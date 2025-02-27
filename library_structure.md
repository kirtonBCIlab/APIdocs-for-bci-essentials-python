# BCI Essentials Python Library Structure

## Overview

The BCI Essentials Python library is a collection of tools and methods for Brain-Computer Interface (BCI) development, analysis, and implementation. The library is organized into multiple modules, each focusing on a specific aspect of BCI processing.

## Library Structure

```
bci_essentials/
│
├── __init__.py                  # Package initialization
│
├── bci_controller.py            # Main controller class for BCI processing
├── signal_processing.py         # Signal processing tools (filters, etc.)
├── channel_selection.py         # Channel selection algorithms
├── session_saving.py            # Functions for saving and loading sessions
├── resting_state.py             # Tools for resting state EEG analysis
│
├── classification/              # Classification algorithms for BCI paradigms
│   ├── __init__.py              # Package initialization
│   ├── generic_classifier.py    # Base class for all classifiers
│   ├── erp_rg_classifier.py     # ERP Riemannian Geometry classifier
│   ├── erp_single_channel_classifier.py  # ERP classifier for single channel data
│   ├── mi_classifier.py         # Motor Imagery classifier
│   ├── null_classifier.py       # Empty classifier (returns all 0s)
│   ├── ssvep_basic_tf_classifier.py  # Training-free SSVEP classifier
│   ├── ssvep_riemannian_mdm_classifier.py  # SSVEP Riemannian MDM classifier
│   └── switch_mdm_classifier.py # Binary switch classifier
│
├── data_tank/                   # Data storage and management
│   ├── __init__.py              # Package initialization
│   └── data_tank.py             # DataTank class for managing EEG data
│
├── io/                          # Input/Output functionality
│   ├── __init__.py              # Package initialization
│   ├── sources.py               # Abstract base classes for data sources
│   ├── messenger.py             # Communication interface between BCI components
│   ├── lsl_sources.py           # Lab Streaming Layer (LSL) implementation of data sources
│   ├── lsl_messenger.py         # LSL implementation of messenger
│   └── xdf_sources.py           # XDF file implementation of data sources
│
├── paradigm/                    # BCI paradigm implementations
│   ├── __init__.py              # Package initialization
│   ├── paradigm.py              # Base class for BCI paradigms
│   ├── mi_paradigm.py           # Motor Imagery paradigm
│   ├── p300_paradigm.py         # P300/ERP paradigm
│   └── ssvep_paradigm.py        # SSVEP paradigm
│
└── utils/                       # Utility functions and classes
    ├── __init__.py              # Package initialization
    ├── logger.py                # Logging functionality
    └── reduce_to_single_channel.py  # Utility for single-channel analysis
```

## Module Descriptions

### Root Level Modules

- `bci_controller.py`: Contains the main `BciController` class that manages EEG data processing, trial management, and classification
- `signal_processing.py`: Provides functions for filtering (bandpass, lowpass, highpass, notch) and signal processing
- `channel_selection.py`: Implements algorithms for selecting optimal channels for BCI performance
- `session_saving.py`: Tools for saving and loading classifier models and sessions
- `resting_state.py`: Functions for analyzing resting state EEG data, including bandpower features and alpha peak detection

### Classification

Various classifier implementations for different BCI paradigms:
- ERP (Event-Related Potential) classifiers
- MI (Motor Imagery) classifiers 
- SSVEP (Steady-State Visual Evoked Potential) classifiers
- Generic and utility classifiers

### Data Tank

Storage management for EEG data:
- `data_tank.py`: Stores raw EEG data, markers, epochs, and provides data retrieval functionality

### IO (Input/Output)

Handles data acquisition and communication:
- `sources.py`: Abstract base classes defining interfaces for EEG and marker data sources
- `messenger.py`: Interface for communication between BCI components and external applications
- `lsl_sources.py`: Lab Streaming Layer (LSL) implementations of EEG and marker sources
- `lsl_messenger.py`: LSL implementation of the messenger interface for sending events
- `xdf_sources.py`: XDF file-based implementations of EEG and marker sources for offline analysis

### Paradigm

Defines BCI experiment paradigms:
- `paradigm.py`: Base class defining common functionality for all paradigms
- `mi_paradigm.py`: Implements the Motor Imagery paradigm for motor movement/imagination tasks
- `p300_paradigm.py`: Implements the P300/ERP paradigm for event-related potential detection
- `ssvep_paradigm.py`: Implements the SSVEP paradigm for steady-state visual evoked potential detection

### Utils

Utility functions used throughout the library:
- `logger.py`: Custom logging functionality
- `reduce_to_single_channel.py`: Tools for working with single-channel data
```