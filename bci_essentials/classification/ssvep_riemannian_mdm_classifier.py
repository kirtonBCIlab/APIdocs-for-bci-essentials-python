# Stock libraries
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances

# Custom libraries
# - Append higher directory to import bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))
from classification.generic_classifier import Generic_classifier
from bci_essentials.visuals import *
from bci_essentials.signal_processing import *
from bci_essentials.channel_selection import *


class SSVEP_riemannian_mdm_classifier(Generic_classifier):
    """
    Classifier SSVEP based on relative band power at the expected frequencies
    """

    def set_ssvep_settings(self, n_splits=3, random_seed=42, n_harmonics=2, f_width=0.2, covariance_estimator="scm"):
        # Build the cross-validation split
        self.n_splits = n_splits
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

        self.rebuild = True

        self.n_harmonics = n_harmonics
        self.f_width = f_width
        self.covariance_estimator = covariance_estimator

        # Use an MDM classifier, maybe there will be other options later
        mdm = MDM(metric=dict(mean='riemann', distance='riemann'), n_jobs = 1)
        self.clf_model = Pipeline([("MDM", mdm)])
        self.clf = Pipeline([("MDM", mdm)])

    def get_ssvep_supertrial(
        self,
        X, 
        target_freqs, 
        fsample, 
        f_width=0.4, 
        n_harmonics=2, 
        covariance_estimator="scm"):
        """Get SSVEP Supertrial

        Creates the Riemannian Geometry supertrial for SSVEP

        Parameters
        ----------
        X : numpy array 
            Windows of EEG data, nwindows X nchannels X nsamples
        target_freqs : numpy array
            Target frequencies for the SSVEP
        fsample : float
            Sampling rate
        f_width : float, optional
            Width of frequency bins to be used around the target frequencies 
            (default 0.4)
        n_harmonics : int, optional
            Number of harmonics to be used for each frequency (default is 2)
        covarianc_estimator : str, optional
            Covariance Estimator (see Covariances - pyriemann) (default "scm")

        Returns
        -------
        super_X : numpy array
            Supertrials of X with the dimensions nwindows 
            by (nchannels*number of target_freqs) 
            by (nchannels*number of target_freqs)
        """
        nwindows, nchannels, nsamples = X.shape
        n_target_freqs = len(target_freqs)

        super_X = np.zeros([nwindows, nchannels*n_target_freqs, 
                            nchannels*n_target_freqs])

        # Create super trial of all trials filtered at all bands
        for w in range(nwindows):
            for tf, target_freq in enumerate(target_freqs):
                lower_bound = int((nchannels*tf))
                upper_bound = int((nchannels*tf)+nchannels)

                signal = X[w,:,:]
                for f in range(n_harmonics):
                    if f == 0:
                        filt_signal = bandpass(signal, 
                                            f_low=target_freq-(f_width/2), 
                                            f_high=target_freq+(f_width/2), 
                                            order=5, 
                                            fsample=fsample)
                    else:
                        filt_signal += bandpass(signal, 
                                            f_low=(target_freq*(f+1))-(f_width/2), 
                                            f_high=(target_freq*(f+1))+(f_width/2), 
                                            order=5, fsample=fsample)

                cov_mat = Covariances(estimator=covariance_estimator).transform(np.expand_dims(filt_signal, axis=0))

                cov_mat_diag = np.diag(np.diag(cov_mat[0,:,:]))

                super_X[w, lower_bound:upper_bound, lower_bound:upper_bound] = cov_mat_diag

        return super_X
    
    def fit(self, print_fit=True, print_performance=True):
        # get dimensions
        X = self.X


        # Convert each window of X into a SPD of dimensions [nwindows, nchannels*nfreqs, nchannels*nfreqs]
        nwindows, nchannels, nsamples = self.X.shape 

        #################
        # Try rebuilding the classifier each time
        if self.rebuild == True:
            self.next_fit_window = 0
            self.clf = self.clf_model

        # get temporal subset
        subX = self.X[self.next_fit_window:,:,:]
        suby = self.y[self.next_fit_window:]
        self.next_fit_window = nwindows

        # Init predictions to all false 
        preds = np.zeros(nwindows)

        def ssvep_kernel(subX, suby):
            for train_idx, test_idx in self.cv.split(subX,suby):
                self.clf = self.clf_model

                X_train, X_test = subX[train_idx], subX[test_idx]
                y_train, y_test = suby[train_idx], suby[test_idx]

                # get the covariance matrices for the training set
                X_train_super = self.get_ssvep_supertrial(X_train, self.target_freqs, fsample=256, n_harmonics=self.n_harmonics, f_width=self.f_width, covariance_estimator=self.covariance_estimator)
                X_test_super = self.get_ssvep_supertrial(X_test, self.target_freqs, fsample=256, n_harmonics=self.n_harmonics, f_width=self.f_width, covariance_estimator=self.covariance_estimator)

                # fit the classsifier
                self.clf.fit(X_train_super, y_train)
                preds[test_idx] = self.clf.predict(X_test_super)

            accuracy = sum(preds == self.y)/len(preds)
            precision = precision_score(self.y,preds, average="micro")
            recall = recall_score(self.y, preds, average="micro")

            model = self.clf

            return model, preds, accuracy, precision, recall

        # Check if channel selection is true
        if self.channel_selection_setup:
            print("Doing channel selection")

            updated_subset, updated_model, preds, accuracy, precision, recall = channel_selection_by_method(ssvep_kernel, self.X, self.y, self.channel_labels,             # kernel setup
                                                                            self.chs_method, self.chs_metric, self.chs_initial_subset,                                      # wrapper setup
                                                                            self.chs_max_time, self.chs_min_channels, self.chs_max_channels, self.chs_performance_delta,    # stopping criterion
                                                                            self.chs_n_jobs, self.chs_output) 
                
            print("The optimal subset is ", updated_subset)

            self.subset = updated_subset
            self.clf = updated_model
        else: 
            print("Not doing channel selection")
            self.clf, preds, accuracy, precision, recall= ssvep_kernel(subX, suby)

        # Print performance stats

        self.offline_window_count = nwindows
        self.offline_window_counts.append(self.offline_window_count)

        # accuracy
        accuracy = sum(preds == self.y)/len(preds)
        self.offline_accuracy.append(accuracy)
        if print_performance:
            print("accuracy = {}".format(accuracy))

        # precision
        precision = precision_score(self.y, preds, average='micro')
        self.offline_precision.append(precision)
        if print_performance:
            print("precision = {}".format(precision))

        # recall
        recall = recall_score(self.y, preds, average='micro')
        self.offline_recall.append(recall)
        if print_performance:
            print("recall = {}".format(recall))

        # confusion matrix in command line
        cm = confusion_matrix(self.y, preds)
        self.offline_cm = cm
        if print_performance:
            print("confusion matrix")
            print(cm)

    def predict(self, X, print_predict=True):
        # if X is 2D, make it 3D with one as first dimension
        if len(X.shape) < 3:
            X = X[np.newaxis, ...]

        X = self.get_subset(X)

        if print_predict:
            print("the shape of X is", X.shape)

        X_super = self.get_ssvep_supertrial(X, self.target_freqs, fsample=256, n_harmonics=self.n_harmonics, f_width=self.f_width)

        pred = self.clf.predict(X_super)
        pred_proba = self.clf.predict_proba(X_super)

        if print_predict:
            print(pred)
            print(pred_proba)

        for i in range(len(pred)):
            self.predictions.append(pred[i])
            self.pred_probas.append(pred_proba[i])

        return pred
