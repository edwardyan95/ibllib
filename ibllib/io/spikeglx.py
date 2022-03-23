import json
import logging
from pathlib import Path
import re
import shutil

import numpy as np
import mtscomp
from iblutil.io import hashfile
from iblutil.util import Bunch
from one.alf.io import remove_uuid_file
from one.api import ONE

from ibllib.ephys import neuropixel as neuropixel

SAMPLE_SIZE = 2  # int16
DEFAULT_BATCH_SIZE = 1e6
_logger = logging.getLogger('ibllib')
# provided as convenience if no meta-data is provided, always better to read from meta
S2V_AP = 2.34375e-06
S2V_LFP = 4.6875e-06


class Reader:
    """
    Class for SpikeGLX reading purposes
    Some format description was found looking at the Matlab SDK here
    https://github.com/billkarsh/SpikeGLX/blob/master/MATLAB-SDK/DemoReadSGLXData.m

    To open a spikeglx file that has an associated meta-data file:
    sr = spikeglx.Reader(bin_file_path)

    To open a flat binary file:

    sr = spikeglx.Reader(bin_file_path, nc=385, ns=nsamples, fs=30000)
    one can provide more options to the reader:
    sr = spikeglx.Reader(..., dtype='int16, s2mv=2.34375e-06)

    usual sample 2 mv conversion factors:
        s2mv = 2.34375e-06 (NP1 ap banc) : default value used
        s2mv = 4.6875e-06 (NP1 lfp band)

    Note: To release system resources the close method must be called
    """

    def __init__(self, sglx_file, open=True, nc=None, ns=None, fs=None, dtype='int16', s2v=None,
                 nsync=None):
        """
        An interface for reading data from a SpikeGLX file
        :param sglx_file: Path to a SpikeGLX file (compressed or otherwise)
        :param open: when True the file is opened
        """
        self.file_bin = Path(sglx_file)
        self.nbytes = self.file_bin.stat().st_size
        self.dtype = np.dtype(dtype)
        file_meta_data = Path(sglx_file).with_suffix('.meta')
        if not file_meta_data.exists():
            err_str = "Instantiating an Reader without meta data requires providing nc, fs and nc parameters"
            assert (nc is not None and fs is not None and nc is not None), err_str
            self.file_meta_data = None
            self.meta = None
            self._nc, self._fs, self._ns = (nc, fs, ns)
            # handles default parameters: if int16 we assume it's a raw recording with 1 sync and sample2mv
            # if its' float32 or something else, we assume the sync channel has been removed and the scaling applied
            if nsync is None:
                nsync = 1 if self.dtype == np.dtype('int16') else 0
            self._nsync = nsync
            if s2v is None:
                s2v = S2V_AP if self.dtype == np.dtype('int16') else 1.0
            self.channel_conversion_sample2v = {'samples': np.ones(nc) * s2v}
            self.channel_conversion_sample2v['samples'][-nsync:] = 1
        else:
            # normal case we continue reading and interpreting the metadata file
            self.file_meta_data = file_meta_data
            self.meta = read_meta_data(file_meta_data)
            self.channel_conversion_sample2v = _conversion_sample2v_from_meta(self.meta)
            self._raw = None
        if open:
            self.open()

    def open(self):
        # if we are not looking at a compressed file, use a memmap, otherwise instantiate mtscomp
        sglx_file = str(self.file_bin)
        if self.is_mtscomp:
            self._raw = mtscomp.Reader()
            self._raw.open(self.file_bin, self.file_bin.with_suffix('.ch'))
            if self._raw.shape != (self.ns, self.nc):
                ftsec = self._raw.shape[0] / self.fs
                self.meta['fileTimeSecs'] = ftsec
                if ftsec > 1:  # avoid the checks for streaming data
                    _logger.warning(f"{sglx_file} : meta data and compressed chunks dont checkout\n"
                                    f"File duration: expected {self.meta['fileTimeSecs']},"
                                    f" actual {ftsec}\n"
                                    f"Will attempt to fudge the meta-data information.")
        else:
            if self.nc * self.ns * self.dtype.itemsize != self.nbytes:
                ftsec = self.file_bin.stat().st_size / self.dtype.itemsize / self.nc / self.fs
                _logger.warning(f"{sglx_file} : meta data and filesize do not checkout\n"
                                f"File size: expected {self.meta['fileSizeBytes']},"
                                f" actual {self.file_bin.stat().st_size}\n"
                                f"File duration: expected {self.meta['fileTimeSecs']},"
                                f" actual {ftsec}\n"
                                f"Will attempt to fudge the meta-data information.")
                self.meta['fileTimeSecs'] = ftsec
            self._raw = np.memmap(sglx_file, dtype=self.dtype, mode='r', shape=(self.ns, self.nc))

    def close(self):
        if self.is_open:
            getattr(self._raw, '_mmap', self._raw).close()

    def __enter__(self):
        if not self.is_open:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getitem__(self, item):
        if isinstance(item, int) or isinstance(item, slice):
            return self.read(nsel=item, sync=False)
        elif len(item) == 2:
            return self.read(nsel=item[0], csel=item[1], sync=False)

    @property
    def sample2volts(self):
        return self.channel_conversion_sample2v[self.type]

    @property
    def geometry(self):
        """
        Gets the geometry, ie. the full trace header for the recording
        :return: dictionary with keys 'row', 'col', 'ind', 'shank', 'adc', 'x', 'y', 'sample_shift'
        """
        return _geometry_from_meta(self.meta)

    @property
    def shape(self):
        return self.ns, self.nc

    @property
    def is_open(self):
        return self._raw is not None

    @property
    def is_mtscomp(self):
        return 'cbin' in self.file_bin.suffix

    @property
    def version(self):
        """Gets the version string: '3A', '3B2', '3B1', 'NP2.1', 'NP2.4'"""
        return None if self.meta is None else _get_neuropixel_version_from_meta(self.meta)

    @property
    def major_version(self):
        """Gets the the major version int: 1 or 2"""
        return None if self.meta is None else _get_neuropixel_major_version_from_meta(self.meta)

    @property
    def rl(self):
        return self.ns / self.fs

    @property
    def type(self):
        """:return: ap, lf or nidq. Useful to index dictionaries """
        if not self.meta:
            return 'samples'
        return _get_type_from_meta(self.meta)

    @property
    def fs(self):
        """ :return: sampling frequency (Hz) """
        return self._fs if self.meta is None else _get_fs_from_meta(self.meta)

    @property
    def nc(self):
        """ :return: number of channels """
        return self._nc if self.meta is None else _get_nchannels_from_meta(self.meta)

    @property
    def nsync(self):
        """:return: number of sync channels"""
        return self._nsync if self.meta is None else len(_get_sync_trace_indices_from_meta(self.meta))

    @property
    def ns(self):
        """ :return: number of samples """
        if self.meta is None:
            return self._ns
        return int(np.round(self.meta.get('fileTimeSecs') * self.fs))

    def read(self, nsel=slice(0, 10000), csel=slice(None), sync=True):
        """
        Read from slices or indexes
        :param slice_n: slice or sample indices
        :param slice_c: slice or channel indices
        :return: float32 array
        """
        if not self.is_open:
            raise IOError('Reader not open; call `open` before `read`')
        darray = self._raw[nsel, csel].astype(np.float32, copy=True)
        darray *= self.channel_conversion_sample2v[self.type][csel]
        if sync:
            return darray, self.read_sync(nsel)
        else:
            return darray

    def read_samples(self, first_sample=0, last_sample=10000, channels=None):
        """
        reads all channels from first_sample to last_sample, following numpy slicing convention
        sglx.read_samples(first=0, last=100) would be equivalent to slicing the array D
        D[:,0:100] where the last axis represent time and the first channels.

         :param first_sample: first sample to be read, python slice-wise
         :param last_sample:  last sample to be read, python slice-wise
         :param channels: slice or numpy array of indices
         :return: numpy array of int16
        """
        if channels is None:
            channels = slice(None)
        return self.read(slice(first_sample, last_sample), channels)

    def read_sync_digital(self, _slice=slice(0, 10000)):
        """
        Reads only the digital sync trace at specified samples using slicing syntax
        >>> sync_samples = sr.read_sync_digital(slice(0,10000))
        """
        if not self.is_open:
            raise IOError('Reader not open; call `open` before `read`')
        if not self.meta:
            _logger.warning('Sync trace not labeled in metadata. Assuming last trace')
        return split_sync(self._raw[_slice, _get_sync_trace_indices_from_meta(self.meta)])

    def read_sync_analog(self, _slice=slice(0, 10000)):
        """
        Reads only the analog sync traces at specified samples using slicing syntax
        >>> sync_samples = sr.read_sync_analog(slice(0,10000))
        """
        if not self.meta:
            return
        csel = _get_analog_sync_trace_indices_from_meta(self.meta)
        if not csel:
            return
        else:
            return self.read(nsel=_slice, csel=csel, sync=False)

    def read_sync(self, _slice=slice(0, 10000), threshold=1.2, floor_percentile=10):
        """
        Reads all sync trace. Convert analog to digital with selected threshold and append to array
        :param _slice: samples slice
        :param threshold: (V) threshold for front detection, defaults to 1.2 V
        :param floor_percentile: 10% removes the percentile value of the analog trace before
         thresholding. This is to avoid DC offset drift
        :return: int8 array
        """
        digital = self.read_sync_digital(_slice)
        analog = self.read_sync_analog(_slice)
        if analog is not None and floor_percentile:
            analog -= np.percentile(analog, 10, axis=0)
        if analog is None:
            return digital
        analog[np.where(analog < threshold)] = 0
        analog[np.where(analog >= threshold)] = 1
        return np.concatenate((digital, np.int8(analog)), axis=1)

    def compress_file(self, keep_original=True, **kwargs):
        """
        Compresses
        :param keep_original: defaults True. If False, the original uncompressed file is deleted
         and the current spikeglx.Reader object is modified in place
        :param kwargs:
        :return: pathlib.Path of the compressed *.cbin file
        """
        file_tmp = self.file_bin.with_suffix('.cbin_tmp')
        assert not self.is_mtscomp
        mtscomp.compress(self.file_bin,
                         out=file_tmp,
                         outmeta=self.file_bin.with_suffix('.ch'),
                         sample_rate=self.fs,
                         n_channels=self.nc,
                         dtype=np.int16,
                         **kwargs)
        file_out = file_tmp.with_suffix('.cbin')
        file_tmp.rename(file_out)
        if not keep_original:
            self.file_bin.unlink()
            self.file_bin = file_out
        return file_out

    def decompress_file(self, keep_original=True, **kwargs):
        """
        Decompresses a mtscomp file
        :param keep_original: defaults True. If False, the original compressed file (input)
        is deleted and the current spikeglx.Reader object is modified in place
        NB: This is not equivalent to overwrite (which replaces the output file)
        :return: pathlib.Path of the decompressed *.bin file
        """
        if 'out' not in kwargs:
            kwargs['out'] = self.file_bin.with_suffix('.bin')
        assert self.is_mtscomp
        r = mtscomp.decompress(self.file_bin, self.file_bin.with_suffix('.ch'), **kwargs)
        r.close()
        if not keep_original:
            self.close()
            self.file_bin.unlink()
            self.file_bin.with_suffix('.ch').unlink()
            self.file_bin = kwargs['out']
        return kwargs['out']

    def verify_hash(self):
        """
        Computes SHA-1 hash and returns True if it matches metadata, False otherwise
        :return: boolean
        """
        if self.is_mtscomp:
            with open(self.file_bin.with_suffix('.ch')) as fid:
                mtscomp_params = json.load(fid)
            sm = mtscomp_params.get('sha1_compressed', None)
            if sm is None:
                _logger.warning("SHA1 hash is not implemented for compressed ephys. To check "
                                "the spikeglx acquisition hash, uncompress the file first !")
                return True
            sm = sm.upper()
        else:
            sm = self.meta.fileSHA1
        sc = hashfile.sha1(self.file_bin).upper()
        if sm == sc:
            log_func = _logger.info
        else:
            log_func = _logger.error
        log_func(f"SHA1 metadata: {sm}")
        log_func(f"SHA1 computed: {sc}")
        return sm == sc


def read(sglx_file, first_sample=0, last_sample=10000):
    """
    Function to read from a spikeglx binary file without instantiating the class.
    Gets the meta-data as well.

    >>> ibllib.io.spikeglx.read('/path/to/file.bin', first_sample=0, last_sample=1000)

    :param sglx_file: full path the the binary file to read
    :param first_sample: first sample to be read, python slice-wise
    :param last_sample: last sample to be read, python slice-wise
    :return: Data array, sync trace, meta-data
    """
    with Reader(sglx_file) as sglxr:
        D, sync = sglxr.read_samples(first_sample=first_sample, last_sample=last_sample)
    return D, sync, sglxr.meta


def read_meta_data(md_file):
    """
    Reads the spkike glx metadata file and parse in a dictionary
    Agnostic: does not make any assumption on the keys/content, it just parses key=values

    :param md_file: last sample to be read, python slice-wise
    :return: Data array, sync trace, meta-data
    """
    with open(md_file) as fid:
        md = fid.read()
    d = {}
    for a in md.splitlines():
        k, v = a.split('=')
        # if all numbers, try to interpret the string
        if v and re.fullmatch('[0-9,.]*', v) and v.count('.') < 2:
            v = [float(val) for val in v.split(',')]
            # scalars should not be nested
            if len(v) == 1:
                v = v[0]
        # tildes in keynames removed
        d[k.replace('~', '')] = v
    d['neuropixelVersion'] = _get_neuropixel_version_from_meta(d)
    d['serial'] = _get_serial_number_from_meta(d)
    return Bunch(d)


def write_meta_data(md, md_file):
    """
    Parses a dict into a spikeglx meta data file
    TODO write a test for this function, (read in, write out and make sure it is the same)
    :param meta: meta data dict
    :param md_file: file to save meta data to
    :return:
    """
    with open(md_file, 'w') as fid:
        for key, val in md.items():
            if isinstance(val, list):
                val = ','.join([str(int(v)) for v in val])
            if isinstance(val, float):
                if val.is_integer():
                    val = int(val)
            fid.write(f'{key}={val}\n')


def _get_savedChans_subset(chns):
    """
    Get the subset of the original channels that are saved per shank
    :param chns:
    :return:
    """
    chn_grps = np.r_[0, np.where(np.diff(chns) != 1)[0] + 1, len(chns)]
    chn_subset = [f'{chns[chn_grps[i]]}:{chns[chn_grps[i + 1] - 1]}'
                  if chn_grps[i] < len(chns) - 1 else f'{chns[chn_grps[i]]}'
                  for i in range(len(chn_grps) - 1)]

    return ','.join([sub for sub in chn_subset])


def _get_serial_number_from_meta(md):
    """
    Get neuropixel serial number from the metadata dictionary
    """
    # imProbeSN for 3A, imDatPrb_sn for 3B2, None for nidq 3B2
    serial = md.get('imProbeSN') or md.get('imDatPrb_sn')
    if serial:
        return int(serial)


def _get_neuropixel_major_version_from_meta(md):
    MAJOR_VERSION = {'3A': 1, '3B2': 1, '3B1': 1, 'NP2.1': 2, 'NP2.4': 2.4}
    version = _get_neuropixel_version_from_meta(md)
    if version is not None:
        return MAJOR_VERSION[version]


def _get_neuropixel_version_from_meta(md):
    """
    Get neuropixel version tag (3A, 3B1, 3B2) from the metadata dictionary
    """
    if 'typeEnabled' in md.keys():
        return '3A'
    prb_type = md.get('imDatPrb_type')
    # Neuropixel 1.0 either 3B1 or 3B2 (ask Olivier about 3B1)
    if prb_type == 0:
        if 'imDatPrb_port' in md.keys() and 'imDatPrb_slot' in md.keys():
            return '3B2'
        else:
            return '3B1'
    # Neuropixel 2.0 single shank
    if prb_type == 21:
        return 'NP2.1'
    # Neuropixel 2.0 four shank
    if prb_type == 24:
        return 'NP2.4'


def _get_sync_trace_indices_from_meta(md):
    """
    Returns a list containing indices of the sync traces in the original array
    """
    typ = _get_type_from_meta(md)
    ntr = int(_get_nchannels_from_meta(md))
    if typ == 'nidq':
        nsync = int(md.get('snsMnMaXaDw')[-1])
    elif typ in ['lf', 'ap']:
        nsync = int(md.get('snsApLfSy')[2])
    return list(range(ntr - nsync, ntr))


def _get_analog_sync_trace_indices_from_meta(md):
    """
    Returns a list containing indices of the sync traces in the original array
    """
    typ = _get_type_from_meta(md)
    if typ != 'nidq':
        return []
    tr = md.get('snsMnMaXaDw')
    nsa = int(tr[-2])
    return list(range(int(sum(tr[0:2])), int(sum(tr[0:2])) + nsa))


def _get_nchannels_from_meta(md):
    return int(md.get('nSavedChans'))


def _get_fs_from_meta(md):
    if md.get('typeThis') == 'imec':
        return md.get('imSampRate')
    else:
        return md.get('niSampRate')


def _get_type_from_meta(md):
    """
    Get neuropixel data type (ap, lf or nidq) from metadata
    """
    snsApLfSy = md.get('snsApLfSy', [-1, -1, -1])
    if snsApLfSy[0] == 0 and snsApLfSy[1] != 0:
        return 'lf'
    elif snsApLfSy[0] != 0 and snsApLfSy[1] == 0:
        return 'ap'
    elif snsApLfSy == [-1, -1, -1] and md.get('typeThis', None) == 'nidq':
        return 'nidq'


def _geometry_from_meta(meta_data):
    """
    Gets the geometry, ie. the full trace header for the recording
    :param meta_data: meta_data dictionary as read by ibllib.io.spikeglx.read_meta_data
    :return: dictionary with keys 'row', 'col', 'ind', 'shank', 'adc', 'x', 'y', 'sample_shift'
    """
    cm = _map_channels_from_meta(meta_data)
    major_version = _get_neuropixel_major_version_from_meta(meta_data)
    if cm is None:
        _logger.warning("Meta data doesn't have geometry (snsShankMap field), returning defaults")
        th = neuropixel.trace_header(version=major_version)
        th['flag'] = th['x'] * 0 + 1.
        return th
    th = cm.copy()
    if major_version == 1:
        # the spike sorting channel maps have a flipped version of the channel map
        th['col'] = - cm['col'] * 2 + 2 + np.mod(cm['row'], 2)
    th.update(neuropixel.rc2xy(th['row'], th['col'], version=major_version))
    th['sample_shift'], th['adc'] = neuropixel.adc_shifts(version=major_version)
    th['ind'] = np.arange(cm['col'].size)
    return th


def _map_channels_from_meta(meta_data):
    """
    Interpret the meta data string to extract an array of channel positions along the shank

    :param meta_data: dictionary output from  spikeglx.read_meta_data
    :return: dictionary of arrays 'shank', 'col', 'row', 'flag', one value per active site
    """
    if 'snsShankMap' in meta_data.keys():
        chmap = re.findall(r'([0-9]*:[0-9]*:[0-9]*:[0-9]*)', meta_data['snsShankMap'])
        # for digital nidq types, the key exists but does not contain any information
        if not chmap:
            return {'shank': None, 'col': None, 'row': None, 'flag': None}
        # shank#, col#, row#, drawflag
        # (nb: drawflag is one should be drawn and considered spatial average)
        chmap = np.array([np.float32(cm.split(':')) for cm in chmap])
        return {k: chmap[:, v] for (k, v) in {'shank': 0, 'col': 1, 'row': 2, 'flag': 3}.items()}


def _conversion_sample2v_from_meta(meta_data):
    """
    Interpret the meta data to extract an array of conversion factors for each channel
    so the output data is in Volts
    Conversion factor is: int2volt / channelGain
    For Lf/Ap interpret the gain string from metadata
    For Nidq, repmat the gains from the trace counts in `snsMnMaXaDw`

    :param meta_data: dictionary output from  spikeglx.read_meta_data
    :return: numpy array with one gain value per channel
    """

    def int2volts(md):
        """ :return: Conversion scalar to Volts. Needs to be combined with channel gains """
        if md.get('typeThis', None) == 'imec':
            if 'imMaxInt' in md:
                return md.get('imAiRangeMax') / int(md['imMaxInt'])
            else:
                return md.get('imAiRangeMax') / 512
        else:
            return md.get('niAiRangeMax') / 32768

    int2volt = int2volts(meta_data)
    version = _get_neuropixel_version_from_meta(meta_data)
    # interprets the gain value from the metadata header:
    if 'imroTbl' in meta_data.keys():  # binary from the probes: ap or lf
        sy_gain = np.ones(int(meta_data['snsApLfSy'][-1]), dtype=np.float32)
        # imroTbl has 384 entries regardless of no of channels saved, so need to index by n_ch
        # TODO need to look at snsSaveChanMap and index channels to get correct gain
        n_chn = _get_nchannels_from_meta(meta_data) - len(_get_sync_trace_indices_from_meta(meta_data))
        if 'NP2' in version:
            # NP 2.0; APGain = 80 for all AP
            # return 0 for LFgain (no LF channels)
            out = {'lf': np.hstack((int2volt / 80 * np.ones(n_chn).astype(np.float32), sy_gain)),
                   'ap': np.hstack((int2volt / 80 * np.ones(n_chn).astype(np.float32), sy_gain))}
        else:
            # the sync traces are not included in the gain values, so are included for
            # broadcast ops
            gain = re.findall(r'([0-9]* [0-9]* [0-9]* [0-9]* [0-9]*)',
                              meta_data['imroTbl'])[:n_chn]
            out = {'lf': np.hstack((np.array([1 / np.float32(g.split(' ')[-1]) for g in gain]) *
                                    int2volt, sy_gain)),
                   'ap': np.hstack((np.array([1 / np.float32(g.split(' ')[-2]) for g in gain]) *
                                    int2volt, sy_gain))}

    # nidaq gain can be read in the same way regardless of NP1.0 or NP2.0
    elif 'niMNGain' in meta_data.keys():  # binary from nidq
        gain = np.r_[
            np.ones(int(meta_data['snsMnMaXaDw'][0], )) / meta_data['niMNGain'] * int2volt,
            np.ones(int(meta_data['snsMnMaXaDw'][1], )) / meta_data['niMAGain'] * int2volt,
            np.ones(int(meta_data['snsMnMaXaDw'][2], )) * int2volt,  # no gain for analog sync
            np.ones(int(np.sum(meta_data['snsMnMaXaDw'][3]), ))]  # no unit for digital sync
        out = {'nidq': gain}

    return out


def split_sync(sync_tr):
    """
    The synchronization channels are stored as single bits, this will split the int16 original
    channel into 16 single bits channels

    :param sync_tr: numpy vector: samples of synchronisation trace
    :return: int8 numpy array of 16 channels, 1 column per sync trace
    """
    sync_tr = np.int16(np.copy(sync_tr))
    out = np.unpackbits(sync_tr.view(np.uint8)).reshape(sync_tr.size, 16)
    out = np.flip(np.roll(out, 8, axis=1), axis=1)
    return np.int8(out)


def get_neuropixel_version_from_folder(session_path):
    ephys_files = glob_ephys_files(session_path, ext='meta')
    return get_neuropixel_version_from_files(ephys_files)


def get_neuropixel_version_from_files(ephys_files):
    if any([ef.get('nidq') for ef in ephys_files]):
        return '3B'
    else:
        return '3A'


def get_probes_from_folder(session_path):
    # should glob the ephys files and get out the labels
    # This assumes the meta files exist on the server (this is the case for now but should it be?)
    ephys_files = glob_ephys_files(session_path, ext='meta')
    probes = []
    for files in ephys_files:
        if files['label']:
            probes.append(files['label'])

    return probes


def glob_ephys_files(session_path, suffix='.meta', ext='bin', recursive=True, bin_exists=True):
    """
    From an arbitrary folder (usually session folder) gets the ap and lf files and labels
    Associated to the subfolders where they are
    the expected folder tree is:
    ├── 3A
    │   ├── imec0
    │   ├── sync_testing_g0_t0.imec0.ap.bin
    │   │   └── sync_testing_g0_t0.imec0.lf.bin
    │   └── imec1
    │      ├── sync_testing_g0_t0.imec1.ap.bin
    │      └── sync_testing_g0_t0.imec1.lf.bin
    └── 3B
        ├── sync_testing_g0_t0.nidq.bin
        ├── imec0
        │   ├── sync_testing_g0_t0.imec0.ap.bin
        │   └── sync_testing_g0_t0.imec0.lf.bin
        └── imec1
            ├── sync_testing_g0_t0.imec1.ap.bin
            └── sync_testing_g0_t0.imec1.lf.bin

    :param bin_exists:
    :param suffix:
    :param ext: file extension to look for, default 'bin' but could also be 'meta' or 'ch'
    :param recursive:
    :param session_path: folder, string or pathlib.Path
    :returns: a list of dictionaries with keys 'ap': apfile, 'lf': lffile and 'label'
    """
    def get_label(raw_ephys_apfile):
        if raw_ephys_apfile.parts[-2] != 'raw_ephys_data':
            return raw_ephys_apfile.parts[-2]
        else:
            return ''

    recurse = '**/' if recursive else ''
    ephys_files = []
    for raw_ephys_file in Path(session_path).glob(f'{recurse}*.ap*{suffix}'):
        raw_ephys_apfile = next(raw_ephys_file.parent.glob(raw_ephys_file.stem + f'.*{ext}'), None)
        if not raw_ephys_apfile and bin_exists:
            continue
        elif not raw_ephys_apfile and ext != 'bin':
            continue
        elif not bin_exists and ext == 'bin':
            raw_ephys_apfile = raw_ephys_file.with_suffix('.bin')
        # first get the ap file
        ephys_files.extend([Bunch({'label': None, 'ap': None, 'lf': None, 'path': None})])
        ephys_files[-1].ap = raw_ephys_apfile
        # then get the corresponding lf file if it exists
        lf_file = raw_ephys_apfile.parent / raw_ephys_apfile.name.replace('.ap.', '.lf.')
        ephys_files[-1].lf = next(lf_file.parent.glob(lf_file.stem + f'.*{ext}'), None)
        # finally, the label is the current directory except if it is bare in raw_ephys_data
        ephys_files[-1].label = get_label(raw_ephys_apfile)
        ephys_files[-1].path = raw_ephys_apfile.parent
    # for 3b probes, need also to get the nidq dataset type
    for raw_ephys_file in Path(session_path).rglob(f'{recurse}*.nidq*{suffix}'):
        raw_ephys_nidqfile = next(raw_ephys_file.parent.glob(raw_ephys_file.stem + f'.*{ext}'),
                                  None)
        if not bin_exists and ext == 'bin':
            raw_ephys_nidqfile = raw_ephys_file.with_suffix('.bin')
        ephys_files.extend([Bunch({'label': get_label(raw_ephys_file),
                                   'nidq': raw_ephys_nidqfile,
                                   'path': raw_ephys_file.parent})])
    return ephys_files


def _mock_spikeglx_file(mock_bin_file, meta_file, ns, nc, sync_depth,
                        random=False, int2volts=0.6 / 32768, corrupt=False):
    """
    For testing purposes, create a binary file with sync pulses to test reading and extraction
    """
    meta_file = Path(meta_file)
    mock_path_bin = Path(mock_bin_file)
    mock_path_meta = mock_path_bin.with_suffix('.meta')
    md = read_meta_data(meta_file)
    assert meta_file != mock_path_meta
    fs = _get_fs_from_meta(md)
    fid_source = open(meta_file)
    fid_target = open(mock_path_meta, 'w+')
    line = fid_source.readline()
    while line:
        line = fid_source.readline()
        if line.startswith('fileSizeBytes'):
            line = f'fileSizeBytes={ns * nc * 2}\n'
        if line.startswith('fileTimeSecs'):
            if corrupt:
                line = f'fileTimeSecs={ns / fs + 1.8324}\n'
            else:
                line = f'fileTimeSecs={ns / fs}\n'
        fid_target.write(line)
    fid_source.close()
    fid_target.close()
    if random:
        D = np.random.randint(-32767, 32767, size=(ns, nc), dtype=np.int16)
    else:  # each channel as an int of chn + 1
        D = np.tile(np.int16((np.arange(nc) + 1) / int2volts), (ns, 1))
        D[0:16, :] = 0
    # the last channel is the sync that we fill with
    sync = np.int16(2 ** np.float32(np.arange(-1, sync_depth)))
    D[:, -1] = 0
    D[:sync.size, -1] = sync
    with open(mock_path_bin, 'w+') as fid:
        D.tofile(fid)
    return {'bin_file': mock_path_bin, 'ns': ns, 'nc': nc, 'sync_depth': sync_depth, 'D': D}


def get_hardware_config(config_file):
    """
    Reads the neuropixel_wirings.json file containing sync mapping and parameters
    :param config_file: folder or json file
    :return: dictionary or None
    """
    config_file = Path(config_file)
    if config_file.is_dir():
        config_file = list(config_file.glob('*.wiring*.json'))
        if config_file:
            config_file = config_file[0]
    if not config_file or not config_file.exists():
        return
    with open(config_file) as fid:
        par = json.loads(fid.read())
    return par


def _sync_map_from_hardware_config(hardware_config):
    """
    :param hardware_config: dictonary from json read of neuropixel_wirings.json
    :return: dictionary where key names refer to object and values to sync channel index
    """
    pin_out = neuropixel.SYNC_PIN_OUT[hardware_config['SYSTEM']]
    sync_map = {hardware_config['SYNC_WIRING_DIGITAL'][pin]: pin_out[pin]
                for pin in hardware_config['SYNC_WIRING_DIGITAL']
                if pin_out[pin] is not None}
    analog = hardware_config.get('SYNC_WIRING_ANALOG')
    if analog:
        sync_map.update({analog[pin]: int(pin[2:]) + 16 for pin in analog})
    return sync_map


def get_sync_map(folder_ephys):
    hc = get_hardware_config(folder_ephys)
    if not hc:
        _logger.warning(f"No channel map for {str(folder_ephys)}")
        return None
    else:
        return _sync_map_from_hardware_config(hc)


def download_raw_partial(url_cbin, url_ch, first_chunk=0, last_chunk=0, one=None, cache_dir=None):
    """
    TODO Document
    :param url_cbin:
    :param url_ch:
    :param first_chunk:
    :param last_chunk:
    :return:
    """
    assert str(url_cbin).endswith('.cbin')
    assert str(url_ch).endswith('.ch')
    one = one or ONE()
    webclient = one.alyx
    cache_dir = cache_dir or webclient.cache_dir
    relpath = Path(url_cbin.replace(webclient._par.HTTP_DATA_SERVER, '.')).parents[0]
    # write the temp file into a subdirectory
    tdir_chunk = f"chunk_{str(first_chunk).zfill(6)}_to_{str(last_chunk).zfill(6)}"
    target_dir = Path(cache_dir, relpath, tdir_chunk)
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # First, download the .ch file if necessary
    if isinstance(url_ch, Path):
        ch_file = url_ch
    else:
        ch_file = Path(webclient.download_file(
            url_ch, target_dir=target_dir, clobber=True, return_md5=False))
        ch_file = remove_uuid_file(ch_file)
    ch_file_stream = target_dir.joinpath(ch_file.name).with_suffix('.stream.ch')

    # Load the .ch file.
    with open(ch_file, 'r') as f:
        cmeta = json.load(f)

    # Get the first sample index, and the number of samples to download.
    i0 = cmeta['chunk_bounds'][first_chunk]
    ns_stream = cmeta['chunk_bounds'][last_chunk + 1] - i0
    total_samples = cmeta['chunk_bounds'][-1]

    # handles the meta file
    meta_local_path = ch_file_stream.with_suffix('.meta')
    if not meta_local_path.exists():
        shutil.copy(ch_file.with_suffix('.meta'), meta_local_path)

    # if the cached version happens to be the same as the one on disk, just load it
    if ch_file_stream.exists():
        with open(ch_file_stream, 'r') as f:
            cmeta_stream = json.load(f)
        if (cmeta_stream.get('chopped_first_sample', None) == i0 and
                cmeta_stream.get('chopped_total_samples', None) == total_samples):
            return Reader(ch_file_stream.with_suffix('.cbin'))
    else:
        shutil.copy(ch_file, ch_file_stream)
    assert ch_file_stream.exists()

    # prepare the metadata file
    cmeta['chunk_bounds'] = cmeta['chunk_bounds'][first_chunk:last_chunk + 2]
    cmeta['chunk_bounds'] = [_ - i0 for _ in cmeta['chunk_bounds']]
    assert len(cmeta['chunk_bounds']) >= 2
    assert cmeta['chunk_bounds'][0] == 0

    first_byte = cmeta['chunk_offsets'][first_chunk]
    cmeta['chunk_offsets'] = cmeta['chunk_offsets'][first_chunk:last_chunk + 2]
    cmeta['chunk_offsets'] = [_ - first_byte for _ in cmeta['chunk_offsets']]
    assert len(cmeta['chunk_offsets']) >= 2
    assert cmeta['chunk_offsets'][0] == 0
    n_bytes = cmeta['chunk_offsets'][-1]
    assert n_bytes > 0

    # Save the chopped chunk bounds and offsets.
    cmeta['sha1_compressed'] = None
    cmeta['sha1_uncompressed'] = None
    cmeta['chopped'] = True
    cmeta['chopped_first_sample'] = i0
    cmeta['chopped_samples'] = ns_stream
    cmeta['chopped_total_samples'] = total_samples

    with open(ch_file_stream, 'w') as f:
        json.dump(cmeta, f, indent=2, sort_keys=True)

    # Download the requested chunks
    cbin_local_path = webclient.download_file(
        url_cbin, chunks=(first_byte, n_bytes),
        target_dir=target_dir, clobber=True, return_md5=False)
    cbin_local_path = remove_uuid_file(cbin_local_path)
    cbin_local_path_renamed = cbin_local_path.with_suffix('.stream.cbin')
    cbin_local_path.replace(cbin_local_path_renamed)
    assert cbin_local_path_renamed.exists()

    reader = Reader(cbin_local_path_renamed)
    return reader
