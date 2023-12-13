"""Module for managing ERP data.

This module provides data classes for different BCI paradigms.

It includes the loading of offline data in `xdf` format
or the live streaming of LSL data.

The loaded/streamed data is added to a buffer such that offline and
online processing pipelines are identical.

Data is pre-processed (using the `signal_processing` module), windowed,
and classified (using one of the `classification` sub-modules).

Classes
-------
- `ErpData` : For processing P300 or other Event Related Potentials
(ERP).

"""

import time
import numpy as np
import matplotlib.pyplot as plt

from .eeg_data import EegData
from .utils.logger import Logger

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


# ERP Data
class ErpData(EegData):
    """
    Class that holds, windows, processes and classifies ERP data.
    """

    # Formats the ERP data, call this every time that a new chunk arrives
    def main(
        self,
        window_start=0.0,
        window_end=0.8,
        eeg_start=0,
        buffer=0.01,
        max_num_options=64,
        max_windows_per_option=50,
        max_windows=10000,
        max_decisions=500,
        max_loops=1000000000,
        training=False,
        train_complete=False,
        online=False,
        pp_type="bandpass",  # Preprocessing method
        pp_low=1,  # Preprocessing: bandpass lower cutoff
        pp_high=40,  # Preprocessing: bandpass upper cutoff
        pp_order=5,  # Preprocessing: bandpass order
        plot_erp=False,
    ):
        """Main function of `ErpData` class.

        Formats the ERP data. Call this every time that a new chunk arrives.

        Runs a while loop that reads in ERP windows from the `ErpData`
        object and processes decision blocks. Can be used in `online` or
        offline mode.
        - If in `online` mode, then the loop will continuously try to read
        in data from the `EegData` object and process it. The loop will
        terminate when `max_loops` is reached, or when manually terminated.
        - If in `offline` mode, then the loop will read in all of the data
        at once, process it, and then terminate.

        Parameters
        ----------
        window_start : float, *optional*
            Start time for ERP sampling window relative to marker (seconds).
            - Default is `0.0`.
        window_end : float, *optional*
            End time for ERP sampling window relative to marker (seconds).
            - Default is `0.8`.
        eeg_start : int, *optional*
            Start time for EEG sampling (seconds).
            - Default is `0`.
        buffer : float, *optional*
            Buffer time for EEG sampling in `online` mode (seconds).
            - Default is `0.01`.
        max_num_options : int, *optional*
            Maximum number of stimulus options (?).
            - Default is `64`.
        max_windows_per_option : int, *optional*
            Maximum number of windows to read in per stimulus option (?).
            - Default is `50`.
        max_windows : int, *optional*
            Maximum number of windows to read in per loop (?).
            - Default is `1000`.
        max_decisions : int, *optional*
            Maximum number of ERP decision blocks to store per loop (?).
            - Default is `500`.
        max_loops : int, *optional*
            Maximum number of loops to run.
            - Default is `1000000`.
        training : bool, *optional*
            Flag to indicate if the data will be used to train a classifier.
            - `True`: The data will be used to train the classifier.
            - `False`: The data will be used to predict with the classifier.
            - Default is `True`.
        train_complete : bool, *optional*
            Flag to indicate if the classifier has been trained.
            - `True`: The classifier has been trained.
            - `False`: The classifier has not been trained.
            - Default is `False`.
        online : bool, *optional*
            Flag to indicate if the data will be processed in `online` mode.
            - `True`: The data will be processed in `online` mode.
            - `False`: The data will be processed in `offline` mode.
            - Default is `True`.
        pp_type : str, *optional*
            Preprocessing method to apply to the EEG data.
            - Default is `"bandpass"`.
        pp_low : int, *optional*
            Low corner frequency for bandpass filter.
            - Default is `1`.
        pp_high : int, *optional*
            Upper corner frequency for bandpass filter.
            - Default is `40`.
        pp_order : int, *optional*
            Order of the bandpass filter.
            - Default is `5`.
        plot_erp : bool, *optional*
            - Default is `False`.

        Returns
        -------
        `None`

        """

        unity_train = True
        unity_label = "null"
        self.num_options = max_num_options

        # plot settings
        self.plot_erp = plot_erp
        if self.plot_erp:
            fig1, axs1 = plt.subplots(self.nchannels)
            fig2, axs2 = plt.subplots(self.nchannels)
            non_target_plot = 99

        # iff this is the first time this function is being called for a given dataset
        if eeg_start == 0:
            self.window_size = window_end - window_start
            self.nsamples = int(np.ceil(self.window_size * self.fsample) + 1)
            self.window_end_buffer = buffer
            self.num_options = max_num_options
            self.max_windows = max_windows
            self.max_windows_per_option = max_windows_per_option
            self.max_decisions = max_decisions

            search_index = 0

            self.window_timestamps = np.arange(self.nsamples) / self.fsample

            # initialize the numbers of markers, windows, and decision blocks to zero
            self.marker_count = 0
            self.nwindows = 0
            self.decision_count = 0

            self.training_labels = np.zeros((self.max_windows), dtype=int)
            self.stim_labels = np.zeros(
                (self.max_windows, self.num_options), dtype=bool
            )
            self.target_index = np.ndarray((self.max_windows), bool)

            # initialize the data structures in numpy arrays
            # ERP windows
            self.erp_windows_raw = np.zeros(
                (self.max_windows, self.nchannels, self.nsamples)
            )
            self.erp_windows_processed = np.zeros(
                (self.max_windows, self.nchannels, self.nsamples)
            )

            # Windows per decision, ie. the number of times each stimulus has flashed
            self.windows_per_decision = np.zeros((self.num_options))

            # Decision blocks are the ensemble averages of all windows collected for each stimulus object
            self.decision_blocks_raw = np.ndarray(
                (self.max_decisions, self.num_options, self.nchannels, self.nsamples)
            )
            self.decision_blocks_processed = np.ndarray(
                (self.max_decisions, self.num_options, self.nchannels, self.nsamples)
            )

            # Big decision blocks contain all decisions, all stimulus objects, all windows, all channels, and all samples (they are BIG)
            self.big_decision_blocks_raw = np.ndarray(
                (
                    self.max_decisions,
                    self.num_options,
                    self.max_windows_per_option,
                    self.nchannels,
                    self.nsamples,
                )
            )
            self.big_decision_blocks_processed = np.ndarray(
                (
                    self.max_decisions,
                    self.num_options,
                    self.max_windows_per_option,
                    self.nchannels,
                    self.nsamples,
                )
            )

            # Initialize the
            self.num_options_per_decision = np.zeros((max_decisions))

            loops = 0

        while loops < max_loops:
            # load data chunk from search start position
            # offline no more data to load
            if online is False:
                loops = max_loops

            # read from sources to get new data
            self._pull_data_from_source()

            # check if there is an available marker, if not, break and wait for more data
            while len(self.marker_timestamps) > self.marker_count:
                loops = 0
                if len(self.marker_data[self.marker_count][0].split(",")) == 1:
                    # if self.marker_data[self.marker_count][0] == 'P300 SingleFlash Begins' or 'P300 SingleFlash Started':
                    if (
                        self.marker_data[self.marker_count][0]
                        == "P300 SingleFlash Started"
                        or self.marker_data[self.marker_count][0]
                        == "P300 SingleFlash Begins"
                        or self.marker_data[self.marker_count][0] == "Trial Started"
                    ):
                        # Note that a marker occured, but do nothing else
                        logger.info("Trial Started")
                        self.marker_count += 1
                        logger.debug(
                            "Increased marker count by 1 to %s", self.marker_count
                        )

                    # once all resting state data is collected then go and compile it
                    elif (
                        self.marker_data[self.marker_count][0]
                        == "Done with all RS collection"
                    ):
                        self._package_resting_state_data()
                        self.marker_count += 1

                    # If training completed then train the classifier
                    elif (
                        self.marker_data[self.marker_count][0] == "Training Complete"
                        and train_complete is False
                    ):
                        if train_complete is False:
                            logger.debug("Training the classifier")
                            self._classifier.fit()
                        train_complete = True
                        training = False
                        self.marker_count += 1
                        # continue

                    # if there is a P300 end flag increment the decision_index by one
                    elif (
                        self.marker_data[self.marker_count][0]
                        == "P300 SingleFlash Ends"
                        or self.marker_data[self.marker_count][0] == "Trial Ends"
                    ):
                        # get the smallest number of windows per decision in the case the are not the same
                        num_ensemble_windows = int(np.min(self.windows_per_decision))

                        # save the number of options
                        self.num_options_per_decision[self.decision_count] = int(
                            self.num_options
                        )

                        # Raw ensemble average
                        ensemble_average_block = np.mean(
                            self.big_decision_blocks_raw[
                                self.decision_count,
                                0 : self.num_options,
                                0:num_ensemble_windows,
                                0 : self.nchannels,
                                0 : self.nsamples,
                            ],
                            axis=1,
                        )
                        self.decision_blocks_raw[
                            self.decision_count,
                            0 : self.num_options,
                            0 : self.nchannels,
                            0 : self.nsamples,
                        ] = ensemble_average_block

                        # Processed ensemble average
                        ensemble_average_block = np.mean(
                            self.big_decision_blocks_processed[
                                self.decision_count,
                                0 : self.num_options,
                                0:num_ensemble_windows,
                                0 : self.nchannels,
                                0 : self.nsamples,
                            ],
                            axis=1,
                        )
                        self.decision_blocks_processed[
                            self.decision_count,
                            0 : self.num_options,
                            0 : self.nchannels,
                            0 : self.nsamples,
                        ] = ensemble_average_block

                        # Reset windows per decision
                        self.windows_per_decision = np.zeros((self.num_options))

                        if self.plot_erp:
                            fig1.show()
                            fig2.show()

                        logger.info(
                            "Marker: %s", self.marker_data[self.marker_count][0]
                        )

                        self.marker_count += 1

                        # ADD IN A CHECK TO MAKE SURE THERE IS SUFFICIENT INFO TO MAKE A DECISION
                        # IF NOT THEN MOVE ON TO THE NEXT ONE
                        if True:
                            # CLASSIFICATION
                            # if the decision block has a label then add to training set
                            # if training:
                            # if self.decision_count <= len(self.labels) - 1:
                            if train_complete is False:
                                # ADD to training set
                                if unity_train:
                                    logger.debug(
                                        "Adding decision block %s to "
                                        + "the classifier with label %s",
                                        self.decision_count,
                                        unity_label,
                                    )
                                    self._classifier.add_to_train(
                                        self.decision_blocks_processed[
                                            self.decision_count,
                                            : self.num_options,
                                            :,
                                            :,
                                        ],
                                        unity_label,
                                    )

                                    # plot what was added
                                    # decision_vis(self.decision_blocks[self.decision_count,:,:,:], self.fsample, unity_label, self.channel_labels)
                                else:
                                    logger.debug(
                                        "Adding decision block %s to "
                                        + "the classifier with label %s",
                                        self.decision_count,
                                        self.labels[self.decision_count],
                                    )
                                    self._classifier.add_to_train(
                                        self.decision_blocks_processed[
                                            self.decision_count,
                                            : self.num_options,
                                            :,
                                            :,
                                        ],
                                        self.labels[self.decision_count],
                                    )

                                    # if the last of the labelled data was just added
                                    if self.decision_count == len(self.labels) - 1:
                                        # FIT
                                        logger.debug("Training the classifier")
                                        self._classifier.fit(n_splits=len(self.labels))

                            # else do the predict the label
                            else:
                                # PREDICT
                                prediction = self._classifier.predict_decision_block(
                                    decision_block=self.decision_blocks_processed[
                                        self.decision_count, 0 : self.num_options, :, :
                                    ],
                                )

                                logger.info(
                                    "%s was selected by the classifier", prediction
                                )

                                if self._messenger is not None:
                                    logger.info("Sending prediction %s", prediction)
                                    self._messenger.prediction(prediction)

                        # TODO: Code is currently unreachable
                        else:
                            logger.error("Insufficient windows to make a decision")
                            self.decision_count -= 1
                            # logger.info(
                            #     "Windows per decision: %s",
                            #     self.windows_per_decision
                            # )

                        self.decision_count += 1
                        self.windows_per_decision = np.zeros((self.num_options))

                        # UPDATE THE SEARCH START LOC
                        # continue
                    else:
                        self.marker_count += 1

                    if online:
                        time.sleep(0.01)
                    loops += 1
                    continue

                # Check if the whole EEG window corresponding to the marker is available
                end_time_plus_buffer = (
                    self.marker_timestamps[self.marker_count] + window_end + buffer
                )

                if self.eeg_timestamps[-1] <= end_time_plus_buffer:
                    # UPDATE THE SEARCH START LOC
                    break

                if self._messenger is not None:
                    marker = self.marker_data[self.marker_count][0]
                    self._messenger.marker_received(marker)

                # Markers are in the format [p300, single (s) or multi (m),num_selections, train_target_index, flash_index_1, flash_index_2, ... ,flash_index_n]
                marker_info = self.marker_data[self.marker_count][0].split(",")

                # unity_flash_indexes
                flash_indices = list()

                for i, info in enumerate(marker_info):
                    # if i == 0:
                    #     bci_string = info
                    if i == 1:
                        self.flash_type = info
                    elif i == 2:
                        # If there is a different number of options
                        if self.num_options != int(info):
                            self.num_options = int(info)

                            # Resize on the first marker
                            self.windows_per_decision = np.zeros((self.num_options))
                    elif i == 3:
                        unity_label = int(info)
                    elif i >= 4:
                        flash_indices.append(int(info))

                self.windows_per_decision[flash_indices] += 1

                # During training,
                # should this be repeated for multiple flash indices
                # for flash_index in flash_indices:
                if training:
                    # Get target info
                    # current_target = target_order[self.decision_count]
                    if unity_train:
                        logger.info("Marker information: %s", marker_info)
                        current_target = unity_label

                    self.training_labels[self.nwindows] = current_target

                    for fi in flash_indices:
                        self.stim_labels[self.nwindows, fi] = True

                    if current_target in flash_indices:
                        self.target_index[self.nwindows] = True
                    else:
                        self.target_index[self.nwindows] = False

                # Find the start time and end time for the window based on the marker timestamp
                start_time = self.marker_timestamps[self.marker_count] + window_start
                # end_time = self.marker_timestamps[self.marker_count] + window_end

                # locate the indices of the window in the eeg data
                for i, s in enumerate(self.eeg_timestamps[search_index:-1]):
                    logger.debug(
                        "Indices (i,s) of the window in the eeg data: (%s,%s)", i, s
                    )
                    if s > start_time:
                        start_loc = search_index + i - 1
                        # if start_loc < 0:
                        #     start_loc = 0

                        break
                end_loc = start_loc + self.nsamples + 1

                # Adjust windows per option
                # self.windows_per_option = np.zeros(self.num_options, dtype=int)

                logger.debug(
                    "Window (start_loc, end_loc): (%s, %s)", start_loc, end_loc
                )
                # linear interpolation and add to numpy array
                for flash_index in flash_indices:
                    for c in range(self.nchannels):
                        eeg_timestamps_adjusted = (
                            self.eeg_timestamps[start_loc:end_loc]
                            - self.eeg_timestamps[start_loc]
                        )

                        channel_data = np.interp(
                            self.window_timestamps,
                            eeg_timestamps_adjusted,
                            self.eeg_data[start_loc:end_loc, c],
                        )

                        # add to raw ERP windows
                        self.erp_windows_raw[
                            self.nwindows, c, 0 : self.nsamples
                        ] = channel_data
                        # self.decision_blocks_raw[self.decision_count, self.nwindows, c, 0:self.nsamples]

                        # if pp_type == "bandpass":
                        #     channel_data_2 = bandpass(channel_data[np.newaxis,:], pp_low, pp_high, pp_order, self.fsample)
                        #     channel_data = channel_data_2[0,:]

                        # # Add to the instance count
                        # self.windows_per_decision[flash_index] += 1

                        if self.plot_erp:
                            if flash_index == current_target:
                                axs1[c].plot(range(self.nsamples), channel_data)

                            elif (
                                non_target_plot == 99 or non_target_plot == flash_index
                            ):
                                axs2[c].plot(range(self.nsamples), channel_data)
                                non_target_plot = flash_index

                        # # add to processed ERP windows
                        # self.erp_windows[self.nwindows, c, 0:self.nsamples] = channel_data

                        # # Does the ensemble avearging
                        # self.decision_blocks[self.decision_count, flash_index, c, 0:self.nsamples] += channel_data

                    # This is where to do preprocessing
                    self.erp_windows_processed[
                        self.nwindows, : self.nchannels, : self.nsamples
                    ] = self._preprocessing(
                        window=self.erp_windows_raw[
                            self.nwindows, : self.nchannels, : self.nsamples
                        ],
                        option=pp_type,
                        order=pp_order,
                        fl=pp_low,
                        fh=pp_high,
                    )

                    # This is where to do artefact rejection
                    self.erp_windows_processed[
                        self.nwindows, : self.nchannels, : self.nsamples
                    ] = self._artefact_rejection(
                        window=self.erp_windows_processed[
                            self.nwindows, : self.nchannels, : self.nsamples
                        ],
                        option=None,
                    )

                    # Add the raw window to the raw decision blocks
                    # self.decision_blocks_raw[self.decision_count, flash_index, 0:self.nchannels, 0:self.nsamples] +=  self.erp_windows_processed
                    self.big_decision_blocks_raw[
                        self.decision_count,
                        flash_index,
                        int(self.windows_per_decision[flash_index] - 1),
                        0 : self.nchannels,
                        0 : self.nsamples,
                    ] = self.erp_windows_raw[
                        self.nwindows, : self.nchannels, : self.nsamples
                    ]

                    self.big_decision_blocks_processed[
                        self.decision_count,
                        flash_index,
                        int(self.windows_per_decision[flash_index] - 1),
                        0 : self.nchannels,
                        0 : self.nsamples,
                    ] = self.erp_windows_processed[
                        self.nwindows, : self.nchannels, : self.nsamples
                    ]

                    # self.windows_per_decision[flash_index] += 1
                # Reset for the next decision

                # iterate to next window
                self.marker_count += 1
                self.nwindows += 1
                search_index = start_loc
                if online:
                    time.sleep(0.000001)

            if online:
                time.sleep(0.000001)
            loops += 1

        # Trim the unused ends of numpy arrays
        if training:
            self.training_labels = self.training_labels[0 : self.nwindows - 1]
            self.target_index = self.target_index[0 : self.nwindows - 1]

        # self.erp_windows = self.erp_windows[0:self.nwindows, 0:self.nchannels, 0:self.nsamples]
        self.erp_windows_raw = self.erp_windows_raw[
            0 : self.nwindows, 0 : self.nchannels, 0 : self.nsamples
        ]
        self.target_index = self.target_index[0 : self.nwindows]
        self.training_labels = self.training_labels[0 : self.nwindows]
        self.stim_labels = self.stim_labels[0 : self.nwindows, :]
        self.num_options_per_decision = self.num_options_per_decision[
            0 : self.decision_count
        ]
        self.decision_blocks_raw = self.decision_blocks_raw[
            0 : self.decision_count, :, 0 : self.nchannels, 0 : self.nsamples
        ]
        self.decision_blocks_processed = self.decision_blocks_processed[
            0 : self.decision_count, :, 0 : self.nchannels, 0 : self.nsamples
        ]
        self.big_decision_blocks_raw = self.big_decision_blocks_raw[
            0 : self.decision_count, :, :, 0 : self.nchannels, 0 : self.nsamples
        ]
        self.big_decision_blocks_processed = self.big_decision_blocks_processed[
            0 : self.decision_count, :, :, 0 : self.nchannels, 0 : self.nsamples
        ]
