import logging
import numpy as np
from pkg_resources import parse_version
from one.alf.io import AlfBunch

import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.io.extractors.training_wheel import Wheel
#from ibllib.io.extractors.camera import CameraTimestampsBpod

_logger = logging.getLogger(__name__)



    
class TrialMod(BaseBpodTrialsExtractor):
    """
    1 is auditory, 0 is visual
    
    """
    save_names = '_ibl_trials_modality.type.npy'
    var_names = 'modality'

    def _extract(self):
        trialmod = np.zeros(len(self.bpod_trials), np.int64)
        for i, t in enumerate(self.bpod_trials):
            trialmod[i] = int(t['trial_type'][2])
        return trialmod

class StimToneFreq(BaseBpodTrialsExtractor):
    """
    only for auditory trials
    
    """
    save_names = '_ibl_trials.stimToneFreq.npy'
    var_names = 'stimToneFreq'

    def _extract(self):
        stimToneFreq = np.zeros(len(self.bpod_trials), np.float64)
        for i, t in enumerate(self.bpod_trials):
            if t['trial_type'][2] == "1":
                stimToneFreq[i] = int(t['stim_tone_freq'])
            else:
                stimToneFreq[i] = np.nan
        return stimToneFreq

class StimPosition(BaseBpodTrialsExtractor):
    """
    only for visual trials
    
    """
    save_names = '_ibl_trials.stimPosition.npy'
    var_names = 'stimPosition'

    def _extract(self):
        stimPosition = np.zeros(len(self.bpod_trials), np.float64)
        for i, t in enumerate(self.bpod_trials):
            if t['trial_type'][2] == "0":
                stimPosition[i] = int(t['position'])
            else:
                stimPosition[i] = np.nan
        return stimPosition






class RewardVolume(BaseBpodTrialsExtractor):
    """
    Load reward volume delivered for each trial.
    **Optional:** saves _ibl_trials.rewardVolume.npy

    Uses reward_current to accumulate the amount of
    """
    save_names = '_ibl_trials.rewardVolume.npy'
    var_names = 'rewardVolume'

    def _extract(self):
        trial_volume = [x['reward_amount'] for x in self.bpod_trials]
        reward_volume = np.array(trial_volume).astype(np.float64)
        assert len(reward_volume) == len(self.bpod_trials)
        return reward_volume


class FeedbackTimes(BaseBpodTrialsExtractor):
    """
    Get the times the water or error tone was delivered to the animal.
    **Optional:** saves _ibl_trials.feedback_times.npy

    Gets reward  and error state init times vectors,
    checks if theintersection of nans is empty, then
    merges the 2 vectors.
    """
    save_names = '_ibl_trials.feedback_times.npy'
    var_names = 'feedback_times'

    @staticmethod
    def get_feedback_times_lt5(session_path, task_collection='raw_behavior_data', data=False):
        if not data:
            data = raw.load_data(session_path, task_collection=task_collection)
        rw_times = [tr['behavior_data']['States timestamps']['reward'][0][0]
                    for tr in data]
        
        assert sum(np.isnan(rw_times)) == 0
        merge = np.array([np.array(times)[~np.isnan(times)] for times in
                          zip(rw_times)]).squeeze()

        return np.array(merge)

    @staticmethod
    def get_feedback_times_ge5(session_path, task_collection='raw_behavior_data', data=False):
        # ger err and no go trig times -- look for BNC2High of trial -- verify
        # only 2 onset times go tone and noise, select 2nd/-1 OR select the one
        # that is grater than the nogo or err trial onset time
        if not data:
            data = raw.load_data(session_path, task_collection=task_collection)
        missed_bnc2 = 0
        rw_times, err_sound_times, merge = [np.zeros([len(data), ]) for _ in range(3)]

        for ind, tr in enumerate(data):
            st = tr['behavior_data']['Events timestamps'].get('BNC2High', None)
            if not st:
                st = np.array([np.nan, np.nan])
                missed_bnc2 += 1
            # xonar soundcard duplicates events, remove consecutive events too close together
            st = np.delete(st, np.where(np.diff(st) < 0.020)[0] + 1)
            rw_times[ind] = tr['behavior_data']['States timestamps']['reward'][0][0]
            # get the error sound only if the reward is nan
            err_sound_times[ind] = st[-1] if st.size >= 2 and np.isnan(rw_times[ind]) else np.nan
        if missed_bnc2 == len(data):
            _logger.warning('No BNC2 for feedback times, filling error trials NaNs')
        merge *= np.nan
        merge[~np.isnan(rw_times)] = rw_times[~np.isnan(rw_times)]
        merge[~np.isnan(err_sound_times)] = err_sound_times[~np.isnan(err_sound_times)]

        return merge

    def _extract(self):
        # Version check
        # if parse_version(self.settings['IBLRIG_VERSION_TAG']) >= parse_version('5.0.0'):
        #     merge = self.get_feedback_times_ge5(self.session_path, task_collection=self.task_collection, data=self.bpod_trials)
        # else:
        #     merge = self.get_feedback_times_lt5(self.session_path, task_collection=self.task_collection, data=self.bpod_trials)
        merge = self.get_feedback_times_lt5(self.session_path, task_collection=self.task_collection, data=self.bpod_trials)
        return np.array(merge)


class Intervals(BaseBpodTrialsExtractor):
    """
    Trial start to trial end. Trial end includes 1 or 2 seconds after feedback,
    (depending on the feedback) and 0.5 seconds of iti.
    **Optional:** saves _ibl_trials.intervals.npy

    Uses the corrected Trial start and Trial end timestamp values form PyBpod.
    """
    save_names = '_ibl_trials.intervals.npy'
    var_names = 'intervals'

    def _extract(self):
        starts = [t['behavior_data']['Trial start timestamp'] for t in self.bpod_trials]
        ends = [t['behavior_data']['Trial end timestamp'] for t in self.bpod_trials]
        return np.array([starts, ends]).T







class RuleCueTriggerTimes(BaseBpodTrialsExtractor):
    """
    Get trigger times of rule Cue from state machine.

    Current software solution for triggering sounds uses PyBpod soft codes.
    Delays can be in the order of 10's of ms. This is the time when the command
    to play the sound was executed. To measure accurate time, either getting the
    sound onset from xonar soundcard sync pulse (latencies may vary).
    """
    save_names = '_ibl_trials.ruleCueTrigger_times.npy'
    var_names = 'ruleCueTrigger_times'

    def _extract(self):
        goCue = np.array([tr['behavior_data']['States timestamps']
                            ['rule_on'][0][0] for tr in self.bpod_trials])
        return goCue


class TrialType(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials_feedback.type.npy'
    var_name = 'trial_feedback_type'

    def _extract(self):
        trial_type = []
        for tr in self.bpod_trials:
            if ~np.isnan(tr["behavior_data"]["States timestamps"]["reward"][0][0]):
                trial_type.append(1)
            elif ~np.isnan(tr["behavior_data"]["States timestamps"]["error"][0][0]):
                trial_type.append(-1)
            elif ~np.isnan(tr["behavior_data"]["States timestamps"]["no_go"][0][0]):
                trial_type.append(0)
            else:
                _logger.warning("Trial is not in set {-1, 0, 1}, appending NaN to trialType")
                trial_type.append(np.nan)
        return np.array(trial_type)


class RuleCueTimes(BaseBpodTrialsExtractor):
    """
    Get trigger times of rule Cue from state machine (high TTL from BNC port2, transmitted from sound amplifier).

    Current software solution for triggering sounds uses PyBpod soft codes.
    Delays can be in the order of 10-100s of ms. This is the time when the command
    to play the sound was executed. To measure accurate time, either getting the
    sound onset from the future microphone OR the new xonar soundcard and
    setup developed by Sanworks guarantees a set latency (in testing).
    """
    save_names = '_ibl_trials.ruleCue_times.npy'
    var_names = 'ruleCue_times'

    def _extract(self):
        rule_cue_times = np.zeros([len(self.bpod_trials), ])
        for ind, tr in enumerate(self.bpod_trials):
            if raw.get_port_events(tr, 'BNC2'):
                bnchigh = tr['behavior_data']['Events timestamps'].get('BNC2High', None)
                if bnchigh:
                    rule_cue_times[ind] = bnchigh[0]
                    continue
                bnclow = tr['behavior_data']['Events timestamps'].get('BNC2Low', None)
                if bnclow:
                    rule_cue_times[ind] = bnclow[0] - 0.1
                    continue
                rule_cue_times[ind] = np.nan
            else:
                rule_cue_times[ind] = np.nan

        nmissing = np.sum(np.isnan(rule_cue_times))
        # Check if all stim_syncs have failed to be detected
        if np.all(np.isnan(rule_cue_times)):
            _logger.warning(
                f'{self.session_path}: Missing ALL !! BNC2 TTLs ({nmissing} trials)')
        # Check if any stim_sync has failed be detected for every trial
        elif np.any(np.isnan(rule_cue_times)):
            _logger.warning(f'{self.session_path}: Missing BNC2 TTLs on {nmissing} trials')

        return rule_cue_times


class IncludedTrials(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.included.npy'
    var_names = 'included'

    def _extract(self):
        if parse_version(self.settings['IBLRIG_VERSION_TAG']) >= parse_version('5.0.0'):
            trials_included = self.get_included_trials_ge5(
                data=self.bpod_trials, settings=self.settings)
        else:
            trials_included = self.get_included_trials_lt5(data=self.bpod_trials)
        return trials_included

    @staticmethod
    def get_included_trials_lt5(data=False):
        trials_included = np.array([True for t in data])
        return trials_included

    @staticmethod
    def get_included_trials_ge5(data=False, settings=False):
        trials_included = np.array([True for t in data])
        if ('SUBJECT_DISENGAGED_TRIGGERED' in settings.keys() and settings[
                'SUBJECT_DISENGAGED_TRIGGERED'] is not False):
            idx = settings['SUBJECT_DISENGAGED_TRIALNUM'] - 1
            trials_included[idx:] = False
        return trials_included







# class StimFreezeTriggerTimes(BaseBpodTrialsExtractor):
#     var_names = 'stimFreezeTrigger_times'

#     def _extract(self):
#         if parse_version(self.settings["IBLRIG_VERSION_TAG"]) < parse_version("6.2.5"):
#             return np.ones(len(self.bpod_trials)) * np.nan
#         freeze_reward = np.array(
#             [
#                 True
#                 if np.all(~np.isnan(tr["behavior_data"]["States timestamps"]["freeze_reward"][0]))
#                 else False
#                 for tr in self.bpod_trials
#             ]
#         )
#         freeze_error = np.array(
#             [
#                 True
#                 if np.all(~np.isnan(tr["behavior_data"]["States timestamps"]["freeze_error"][0]))
#                 else False
#                 for tr in self.bpod_trials
#             ]
#         )
#         no_go = np.array(
#             [
#                 True
#                 if np.all(~np.isnan(tr["behavior_data"]["States timestamps"]["no_go"][0]))
#                 else False
#                 for tr in self.bpod_trials
#             ]
#         )
#         assert (np.sum(freeze_error) + np.sum(freeze_reward) +
#                 np.sum(no_go) == len(self.bpod_trials))
#         stimFreezeTrigger = np.array([])
#         for r, e, n, tr in zip(freeze_reward, freeze_error, no_go, self.bpod_trials):
#             if n:
#                 stimFreezeTrigger = np.append(stimFreezeTrigger, np.nan)
#                 continue
#             state = "freeze_reward" if r else "freeze_error"
#             stimFreezeTrigger = np.append(
#                 stimFreezeTrigger, tr["behavior_data"]["States timestamps"][state][0][0]
#             )
#         return stimFreezeTrigger


# class StimOffTriggerTimes(BaseBpodTrialsExtractor):
#     var_names = 'stimOffTrigger_times'

#     def _extract(self):
#         if parse_version(self.settings["IBLRIG_VERSION_TAG"]) >= parse_version("6.2.5"):
#             stim_off_trigger_state = "hide_stim"
#         elif parse_version(self.settings["IBLRIG_VERSION_TAG"]) >= parse_version("5.0.0"):
#             stim_off_trigger_state = "exit_state"
#         else:
#             stim_off_trigger_state = "trial_start"

#         stimOffTrigger_times = np.array(
#             [tr["behavior_data"]["States timestamps"][stim_off_trigger_state][0][0]
#              for tr in self.bpod_trials]
#         )
#         # If pre version 5.0.0 no specific nogo Off trigger was given, just return trial_starts
#         if stim_off_trigger_state == "trial_start":
#             return stimOffTrigger_times

#         no_goTrigger_times = np.array(
#             [tr["behavior_data"]["States timestamps"]["no_go"][0][0] for tr in self.bpod_trials]
#         )
#         # Stim off trigs are either in their own state or in the no_go state if the
#         # mouse did not move, if the stim_off_trigger_state always exist
#         # (exit_state or trial_start)
#         # no NaNs will happen, NaNs might happen in at last trial if
#         # session was stopped after response
#         # if stim_off_trigger_state == "hide_stim":
#         #     assert all(~np.isnan(no_goTrigger_times) == np.isnan(stimOffTrigger_times))
#         # Patch with the no_go states trig times
#         stimOffTrigger_times[~np.isnan(no_goTrigger_times)] = no_goTrigger_times[
#             ~np.isnan(no_goTrigger_times)
#         ]
#         return stimOffTrigger_times


class StimOnTriggerTimes(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.stimOnTrigger_times.npy'
    var_names = 'stimOnTrigger_times'

    def _extract(self):
        # Get the stim_on_state that triggers the onset of the stim
        stim_on_state = np.array([tr['behavior_data']['States timestamps']
                                 ['stim_on'][0] for tr in self.bpod_trials])
        return stim_on_state[:, 0].T


class StimOnTimes_deprecated(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.stimOn_times.npy'
    var_names = 'stimOn_times'

    def _extract(self):
        """
        Find the time of the state machine command to turn on the stim
        (state stim_on start or rotary_encoder_event2)
        Find the next frame change from the photodiode after that TS.
        Screen is not displaying anything until then.
        (Frame changes are in BNC1 High and BNC1 Low)
        """
        # Version check
        _logger.warning("Deprecation Warning: this is an old version of stimOn extraction."
                        "From version 5., use StimOnOffFreezeTimes")
        if parse_version(self.settings['IBLRIG_VERSION_TAG']) >= parse_version('5.0.0'):
            stimOn_times = self.get_stimOn_times_ge5(self.session_path, data=self.bpod_trials,
                                                     task_collection=self.task_collection)
        else:
            stimOn_times = self.get_stimOn_times_lt5(self.session_path, data=self.bpod_trials,
                                                     task_collection=self.task_collection)
        return np.array(stimOn_times)

    @staticmethod
    def get_stimOn_times_ge5(session_path, data=False, task_collection='raw_behavior_data'):
        """
        Find first and last stim_sync pulse of the trial.
        stimOn_times should be the first after the stim_on state.
        (Stim updates are in BNC1High and BNC1Low - frame2TTL device)
        Check that all trials have frame changes.
        Find length of stim_on_state [start, stop].
        If either check fails the HW device failed to detect the stim_sync square change
        Substitute that trial's missing or incorrect value with a NaN.
        return stimOn_times
        """
        if not data:
            data = raw.load_data(session_path, task_collection=task_collection)
        # Get all stim_sync events detected
        stim_sync_all = [raw.get_port_events(tr, 'BNC1') for tr in data]
        stim_sync_all = [np.array(x) for x in stim_sync_all]
        # Get the stim_on_state that triggers the onset of the stim
        stim_on_state = np.array([tr['behavior_data']['States timestamps']
                                 ['stim_on'][0] for tr in data])

        stimOn_times = np.array([])
        for sync, on, off in zip(
                stim_sync_all, stim_on_state[:, 0], stim_on_state[:, 1]):
            pulse = sync[np.where(np.bitwise_and((sync > on), (sync <= off)))]
            if pulse.size == 0:
                stimOn_times = np.append(stimOn_times, np.nan)
            else:
                stimOn_times = np.append(stimOn_times, pulse)

        nmissing = np.sum(np.isnan(stimOn_times))
        # Check if all stim_syncs have failed to be detected
        if np.all(np.isnan(stimOn_times)):
            _logger.error(f'{session_path}: Missing ALL BNC1 TTLs ({nmissing} trials)')

        # Check if any stim_sync has failed be detected for every trial
        if np.any(np.isnan(stimOn_times)):
            _logger.warning(f'{session_path}: Missing BNC1 TTLs on {nmissing} trials')

        return stimOn_times

    @staticmethod
    def get_stimOn_times_lt5(session_path, data=False, task_collection='raw_behavior_data'):
        """
        Find the time of the statemachine command to turn on hte stim
        (state stim_on start or rotary_encoder_event2)
        Find the next frame change from the photodiodeafter that TS.
        Screen is not displaying anything until then.
        (Frame changes are in BNC1High and BNC1Low)
        """
        if not data:
            data = raw.load_data(session_path, task_collection=task_collection)
        stim_on = []
        bnc_h = []
        bnc_l = []
        for tr in data:
            stim_on.append(tr['behavior_data']['States timestamps']['stim_on'][0][0])
            if 'BNC1High' in tr['behavior_data']['Events timestamps'].keys():
                bnc_h.append(np.array(tr['behavior_data']
                                      ['Events timestamps']['BNC1High']))
            else:
                bnc_h.append(np.array([np.NINF]))
            if 'BNC1Low' in tr['behavior_data']['Events timestamps'].keys():
                bnc_l.append(np.array(tr['behavior_data']
                                      ['Events timestamps']['BNC1Low']))
            else:
                bnc_l.append(np.array([np.NINF]))

        stim_on = np.array(stim_on)
        bnc_h = np.array(bnc_h, dtype=object)
        bnc_l = np.array(bnc_l, dtype=object)

        count_missing = 0
        stimOn_times = np.zeros_like(stim_on)
        for i in range(len(stim_on)):
            hl = np.sort(np.concatenate([bnc_h[i], bnc_l[i]]))
            stot = hl[hl > stim_on[i]]
            if np.size(stot) == 0:
                stot = np.array([np.nan])
                count_missing += 1
            stimOn_times[i] = stot[0]

        if np.all(np.isnan(stimOn_times)):
            _logger.error(f'{session_path}: Missing ALL BNC1 TTLs ({count_missing} trials)')

        if count_missing > 0:
            _logger.warning(f'{session_path}: Missing BNC1 TTLs on {count_missing} trials')

        return np.array(stimOn_times)




class PhasePosQuiescence(BaseBpodTrialsExtractor):
    """Extracts stimulus phase, position and quiescence from Bpod data.
    For extraction of pre-generated events, use the ProbaContrasts extractor instead.
    """
    save_names = (None, None,)
    var_names = ('phase', 'position',)

    def _extract(self, **kwargs):
        phase = np.array([t['stim_phase'] for t in self.bpod_trials])
        position = np.array([t['position'] for t in self.bpod_trials])
        return phase, position




class TrialsTable(BaseBpodTrialsExtractor):
    """
    Extracts the following into a table from Bpod raw data:
        intervals, goCue_times, response_times, choice, stimOn_times, contrastLeft, contrastRight,
        feedback_times, feedbackType, rewardVolume, probabilityLeft, firstMovement_times
    Additionally extracts the following wheel data:
        wheel_timestamps, wheel_position, wheel_moves_intervals, wheel_moves_peak_amplitude
    """
    save_names = ('_ibl_trials.table.pqt',)
    var_names = ('table',)

    def _extract(self, extractor_classes=None, **kwargs):
        base = [Intervals, RuleCueTimes, RuleCueTriggerTimes, StimOnTriggerTimes, StimOnTimes_deprecated, FeedbackTimes, 
                RewardVolume,  TrialMod, StimToneFreq, StimPosition]
        out, _ = run_extractor_classes(
            base, session_path=self.session_path, bpod_trials=self.bpod_trials, settings=self.settings, save=False,
            task_collection=self.task_collection)
        table = AlfBunch({k: v for k, v in out.items() if k not in self.var_names})
        #print(len(table.keys()))
        #assert len(table.keys()) == 9

        return table.to_df(), *(out.pop(x) for x in self.var_names if x != 'table')


def extract_all(session_path, save=False, bpod_trials=None, settings=None, task_collection='raw_behavior_data', save_path=None):
    """Extract trials and wheel data.

    For task versions >= 5.0.0, outputs wheel data and trials.table dataset (+ some extra datasets)

    Parameters
    ----------
    session_path : str, pathlib.Path
        The path to the session
    save : bool
        If true save the data files to ALF
    bpod_trials : list of dicts
        The Bpod trial dicts loaded from the _iblrig_taskData.raw dataset
    settings : dict
        The Bpod settings loaded from the _iblrig_taskSettings.raw dataset

    Returns
    -------
    A list of extracted data and a list of file paths if save is True (otherwise None)
    """
    if not bpod_trials:
        bpod_trials = raw.load_data(session_path, task_collection=task_collection)
    if not settings:
        settings = raw.load_settings(session_path, task_collection=task_collection)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    base = [RuleCueTriggerTimes]
    # Version check
    if parse_version(settings['IBLRIG_VERSION_TAG']) >= parse_version('5.0.0'):
        # We now extract a single trials table
        base.extend([
            StimOnTriggerTimes, 
            TrialsTable, PhasePosQuiescence, TrialMod, StimToneFreq, StimPosition
        ])
    else:
        base.extend([
            Intervals, IncludedTrials,
            StimOnTimes_deprecated, RewardVolume, FeedbackTimes, RuleCueTimes, PhasePosQuiescence, TrialMod
        ])

    out, fil = run_extractor_classes(base, save=save, session_path=session_path, bpod_trials=bpod_trials, settings=settings,
                                     task_collection=task_collection, path_out=save_path)
    return out, fil
