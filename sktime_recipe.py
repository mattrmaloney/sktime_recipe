# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:31:46 2023

@author: mmaloney
"""
from sktime.forecasting.compose import TransformedTargetForecaster,ForecastX,ForecastingPipeline, ColumnEnsembleForecaster
from sktime.transformations.compose import OptionalPassthrough,ColumnwiseTransformer,TransformerPipeline, FeatureUnion, Id
from sktime.transformations.series.subset import ColumnSelect
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.hierarchical.reconcile import Reconciler
from sktime.datatypes import check_raise
from datetime import timedelta
import numpy as np
import pandas as pd
import itertools
import copy
import types

class XTranspipeSpec:
    """
    Transformer pipelines for X are more complex because they can be applied 
    to different column subsets. This custom class assists with defining them.
    """

    def __init__(self):
        """Initialize the transformer pipeline specification."""
        self.X_transpipe = None
        self.X_trans_hyper_search_space = None
        self.X_trans_count = 0

    def add_X_transform(self, transform, trans_cols=None, hyper_search_space=None):
        """
        Adds a transformation to the pipeline.

        Parameters:
        - transform (tuple): A tuple where the second element may be an 
          OptionalPassthrough transformer.
        - trans_cols (str or list, optional): Specifies which columns to apply 
          the transformation to. Can be 'All', 'Known', 'Unknown', or a list 
          of column names. Defaults to None.
        - hyper_search_space (dict, optional): Dictionary of hyperparameter 
          search space for the transformation. Defaults to an empty dictionary.
        """

        if hyper_search_space is None:
            hyper_search_space = {}

        # Handle OptionalPassthrough transformation
        if isinstance(transform[1], OptionalPassthrough):
            new_transform = (transform[0], OptionalPassthrough(transformer=transform[1].transformer))
        else:
            new_transform = (transform[0], transform[1])

        # Define transformer pipeline
        if trans_cols == 'All':
            new_transform_pipe = TransformerPipeline(steps=[new_transform])
        else:
            new_transform_pipe = TransformerPipeline(steps=[
                ColumnSelect(columns=trans_cols),
                new_transform
            ])

        # Assign a unique prefix to the pipeline
        trans_pipe_prefix = f'TransformerPipeline_{self.X_trans_count}'

        # Append or initialize transformation pipeline
        if self.X_trans_count == 0:
            self.X_transpipe = [(trans_pipe_prefix, new_transform_pipe)]
        else:
            self.X_transpipe.append((trans_pipe_prefix, new_transform_pipe))

        # Update hyperparameter search space
        if self.X_trans_hyper_search_space is None:
            self.X_trans_hyper_search_space = {
                f'{trans_pipe_prefix}__{k}': v for k, v in hyper_search_space.items()
                }
        else:
            self.X_trans_hyper_search_space.update({
                f'{trans_pipe_prefix}__{k}': v for k, v in hyper_search_space.items()
        })

        # Increment transformation count
        self.X_trans_count += 1

    def update_columns_from_recipe(self, recipe):
        """
        Updates all transformations applied to 'Unknown'
        and 'Known' columns with recipe values

        Parameters:
        - recipe: a recipe object
        """
        for i, (prefix, pipeline) in enumerate(self.X_transpipe):
            steps = pipeline.steps
            if isinstance(steps[0], ColumnSelect) and steps[0].columns == 'Unknown':
                self.X_transpipe[i] = (prefix, TransformerPipeline(steps=[
                    ColumnSelect(columns=recipe.X_unknown),
                    steps[1]  # Keep the original transform
                ]))
            elif isinstance(steps[0], ColumnSelect) and steps[0].columns == 'Known':
                self.X_transpipe[i] = (prefix, TransformerPipeline(steps=[
                    ColumnSelect(columns=recipe.X_known),
                    steps[1]  # Keep the original transform
                ]))


class ForecastingRecipe:
    """
    Class for defining forecasting pipelines, including transformation steps 
    and hyperparameter tuning.
    """

    def __init__(
        self, 
        name, 
        target, 
        forecast_horizon, 
        ts_freq, 
        X_known=None, 
        X_unknown=None,
        aggregate=False, 
        reconciler_method=None, 
        data_importer=None, 
        data_preprocessor=None, 
        known_feature_creator=None
    ):
        """
        Initializes the ForecastingRecipe class.

        Parameters:
        - name (str): Name of the forecasting recipe.
        - target (str): Target variable for forecasting.
        - forecast_horizon (int): Forecasting horizon.
        - ts_freq (str): Time series frequency.
        - X_known (list, optional): List of known features (default: []).
        - X_unknown (list, optional): List of unknown features (default: []).
        - aggregate (bool, optional): Whether to aggregate data (default: False).
        - reconciler_method (str, optional): Method for hierarchical reconciliation.
        - data_importer (callable, optional): Function to import data.
        - data_preprocessor (callable, optional): Function to preprocess data.
        - known_feature_creator (callable, optional): Function to create known features.
        """

        # General attributes
        self.name = name
        self.target = target
        self.forecast_horizon = np.arange(1, forecast_horizon)
        self.ts_freq = ts_freq
        self.X_known = X_known if X_known is not None else []
        self.X_unknown = X_unknown if X_unknown is not None else []

        # y pipeline components
        self.y_core_forecaster = None
        self.y_core_forecaster_hyper_search_space = {}
        self.y_trans_hyper_search_space = {}
        self.y_transpipe = [('id', Id())]  # Default identity transform

        # X pipeline components
        self.X_core_forecaster = None
        self.X_core_forecaster_hyper_search_space = {}
        self.X_transpipe = [('id', Id())]  # Default identity transform
        self.X_trans_hyper_search_space = {}

        # Full pipeline attributes
        self.forecaster = None
        self.hyperparam_search_space = None

        # Custom function attributes (data importer, preprocessor, known feature creator)
        if data_importer:
            self.data_importer = types.MethodType(data_importer, self)

        if data_preprocessor:
            self.data_preprocessor = types.MethodType(data_preprocessor, self)

        if known_feature_creator:
            self.known_feature_creator = types.MethodType(known_feature_creator, self)

        # Aggregation setting
        self.aggregate = aggregate

        # Reconciler method for hierarchical forecasting
        self.reconciler_method = reconciler_method

    def clear_attribute(self, name):
        """
        Resets the specified attribute to its default value.

        Parameters:
        - name (str): Name of the attribute to reset.

        Valid attributes:
        - 'X_transpipe' and 'y_transpipe' -> Reset to an empty list.
        - 'y_trans_hyper_search_space' and 'X_trans_hyper_search_space' -> 
            Reset to an empty dictionary.
        - Any other attribute -> Reset to None.
        """

        if name in ('X_transpipe', 'y_transpipe'):
            setattr(self, name, [])
        elif name in ('y_trans_hyper_search_space', 'X_trans_hyper_search_space'):
            setattr(self, name, {})
        else:
            setattr(self, name, None)
            
    def import_data(self):
        """
        Placeholder function for data import.

        This method should be overridden by assigning a user-defined 
        data importer function to `self.data_importer`.
        """
        raise NotImplementedError("Import function is not defined.")
    
    def data_preprocessor(self, data):
        """
        Placeholder function for data preprocessing.

        This method should be overridden by assigning a user-defined 
        data preprocessor function to `self.data_preprocessor`.
        """
        raise NotImplementedError("Data preprocessor is not defined.")
 
    def known_feature_creator(self, X_index):
        """
        Placeholder function for creating known features.

        This method should be overridden by assigning a user-defined 
        feature creation function to `self.known_feature_creator`.
        """
        raise NotImplementedError("Known feature creator is not defined.")

    def set_create_known_features_fcn(self, fcn):
        """
        Sets a user-defined function for creating known features.

        This method binds the provided function to the instance, allowing it 
        to be used as `self.known_feature_creator`.

        Parameters:
        - fcn (callable): A function that generates known features given an index.
        """
        self.known_feature_creator = types.MethodType(fcn, self)
        
    def update_y_core_forecaster_attributes(self, core_forecaster_tuple):
        """
        Updates the attributes related to the core y forecaster.

        Parameters:
        - core_forecaster_tuple (tuple): A tuple containing:
            - core_forecaster (object): The main y forecaster model.
            - hyper_search_space (dict): Hyperparameter search space for the forecaster.
        """
        self.y_core_forecaster, self.y_core_forecaster_hyper_search_space = core_forecaster_tuple

    def update_y_transpipe_attributes(self, transform_tuple):
        """
        Updates the attributes related to the y transformation pipeline.

        Parameters:
        - transform_tuple (tuple): A tuple containing:
            - y_transpipe (list): A list of transformation steps applied to y.
            - y_trans_hyper_search_space (dict): Hyperparameter search space for y transformations.
        """
        self.y_transpipe, self.y_trans_hyper_search_space = transform_tuple
        
    def update_X_core_forecaster_attributes(self, core_forecaster_tuple):
        """
        Updates the attributes related to the core X forecaster.

        Parameters:
        - core_forecaster_tuple (tuple): A tuple containing:
            - X_core_forecaster (object): The main X forecaster model.
            - X_core_forecaster_hyper_search_space (dict): Hyperparameter search space for the X forecaster.
        """
        self.X_core_forecaster, self.X_core_forecaster_hyper_search_space = core_forecaster_tuple
        
    def update_X_transpipe_attributes(self, transpipe_spec):
        """
        Updates the attributes related to the X transformation pipeline.

        Parameters:
        - transpipe_spec (object): An instance of a transformation pipeline specification
          that contains:
            - X_transpipe (list): List of transformation steps applied to X.
            - X_trans_hyper_search_space (dict): Hyperparameter search space for X transformations.
        """
        # replace 'Unknown', 'Known' with actual column lists
        transpipe_spec.update_columns_from_recipe(recipe=self)

        self.X_transpipe = transpipe_spec.X_transpipe
        self.X_trans_hyper_search_space = transpipe_spec.X_trans_hyper_search_space
  
    def add_known_features(self, X, add_knownX_for_forecast_period=False):
        """
        Adds known features to the dataset.

        This method applies the known feature creation function to the dataset 
        index and merges the generated features with `X`.

        Parameters:
        - X (pd.DataFrame): The input dataset with index corresponding to time steps.
        - add_knownX_for_forecast_period (bool, optional): 
          Whether to extend known features for future forecast periods. Defaults to False.

        Returns:
        - pd.DataFrame: The dataset including known features.
        """
        
        if add_knownX_for_forecast_period==False:
            X_known = self.known_feature_creator(X.index)
            # needs to return a period index # Add automatic conversion here

            if X_known is not None:
                X_out = pd.merge(
                    X_known,
                    X,
                    how='left',
                    left_index = True,
                    right_index = True
                    )
            else:
                X_out = X

        # Handle multi-index case for forecasting periods
        if isinstance(X.index, pd.MultiIndex):
            ts_index = pd.period_range(
                freq=X.index.levels[-1].freq,
                start=min(X.index.levels[-1]),
                periods=len(X.index.levels[-1]) + max(self.forecast_horizon),
            )

            non_dt_levels = np.arange(len(X.index.levels) - 1)
            level_combinations = [X.index.levels[i].unique().tolist() for i in non_dt_levels]
            all_combinations = list(itertools.product(*level_combinations, ts_index))

            df_combinations = pd.DataFrame(all_combinations,
                                           columns=X.index.names).set_index(X.index.names)
            X_known = self.known_feature_creator(df_combinations.index)

            if X_known is not None:
                X_known_multi = pd.merge(X_known,
                                         df_combinations,
                                         how='left',
                                         left_index=True,
                                         right_index=True)
            else:
                X_known_multi = df_combinations
            X_out = pd.merge(X_known_multi, X, how='left', left_index=True, right_index=True)

        else:
            ts_index = pd.period_range(
                freq=X.index.freq,
                start=min(X.index),
                periods=len(X.index) + max(self.forecast_horizon),
            )

            X_known = self.known_feature_creator(ts_index)
            
            if X_known is not None:
                X_out = pd.merge(
                    X_known,
                    X,
                    how='left',
                    left_index = True,
                    right_index = True
                    )
            else:
                X_out = X

        return X_out[self.X_unknown + self.X_known]

    def preprocess_data(self, df, start_dt, add_knownX_for_forecast_period=False,
                        adjust=True, **kwargs):
        """
        Preprocesses the input dataset by applying transformations, filtering, 
        and optionally adding known features.

        Parameters:
        - df (pd.DataFrame): The input dataset containing time series data.
        - start_dt (pd.Timestamp or comparable): The earliest date to include in the dataset.
        - add_knownX_for_forecast_period (bool, optional): Whether to extend known 
          features for forecast periods. Defaults to False.
        - adjust (bool, optional): Whether to apply known feature adjustments. Defaults to True.
        - **kwargs: Additional arguments for the preprocessing function.

        Returns:
        - tuple: (y, X) where:
            - y (pd.DataFrame): Processed target variable data.
            - X (pd.DataFrame): Processed feature data.
        """

        # Handle empty dataframe case
        if df.empty:
            return pd.DataFrame(columns=[self.target] + self.X_unknown), pd.DataFrame()

        # Apply preprocessing function
        out_df = self.data_preprocessor(data=df)

        # Apply aggregation if specified
        if self.aggregate:
            out_df = Aggregator().fit_transform(out_df)

        # Extract target variable (y) and feature data (X)
        y = out_df[[self.target]]
        X = out_df[self.X_unknown]

        # Add known features if specified
        if adjust:
            X = self.add_known_features(X, add_knownX_for_forecast_period)

        # Filter data based on start date
        if isinstance(X.index, pd.MultiIndex):
            ts_ind_name = X.index.levels[-1].name

            X = X[X.index.get_level_values(ts_ind_name) >= start_dt]
            X.index = X.index.remove_unused_levels()

            y = y[y.index.get_level_values(ts_ind_name) >= start_dt]
            y.index = y.index.remove_unused_levels()

        else:
            X = X[X.index >= start_dt]
            y = y[y.index >= start_dt]

        return y, X

    def build_full_pipeline(self):
        """
        Constructs the full forecasting pipeline by integrating y and X pipelines.

        This method builds:
        - `transformed_forecaster_y`: A pipeline for transforming and forecasting y.
        - `transformed_forecaster_X`: A pipeline for transforming X features (if applicable).
        - `self.forecaster`: The final pipeline integrating X and y models.
        - `self.hyperparam_search_space`: The hyperparameter search space for tuning.

        Returns:
        - None: Updates `self.forecaster` and `self.hyperparam_search_space`.
        """

        # Build y transformation pipeline
        transformed_forecaster_y = TransformedTargetForecaster(
            self.y_transpipe + [('forecaster', self.y_core_forecaster)]
        )

        # If X features exist, integrate them
        if (len(self.X_known) + len(self.X_unknown)) > 0:

            if self.X_core_forecaster is not None and len(self.X_unknown) > 0:
                transformed_forecaster_X = FeatureUnion(self.X_transpipe) ** self.X_core_forecaster

                # Joint forecaster integrating X and y
                self.forecaster = ForecastX(
                    forecaster_y=transformed_forecaster_y,
                    forecaster_X=transformed_forecaster_X,
                    columns=self.X_unknown,
                    fit_behaviour='use_forecast',
                    predict_behaviour='use_forecasts'
                )
                
                # Define hyperparameter search space
                self.hyperparam_search_space = {
                    **{f'forecaster_X__FeatureUnion__{k}': v for k, v in self.X_trans_hyper_search_space.items()},
                    **{f'forecaster_y__{k}': v for k, v in self.y_trans_hyper_search_space.items()},
                    **{f'forecaster_X__forecaster__{k}': v for k, v in self.X_core_forecaster_hyper_search_space.items()},
                    **{f'forecaster_y__forecaster__{k}': v for k, v in self.y_core_forecaster_hyper_search_space.items()}
                }

            else:
                # If only y forecaster is used
                self.forecaster = ForecastingPipeline([
                    ('transforms', TransformerPipeline(steps=self.X_transpipe)),
                    ('forecaster', transformed_forecaster_y)
                ])

                # Define hyperparameter search space
                self.hyperparam_search_space = {
                    **{f'transforms__{k}': v for k, v in self.X_trans_hyper_search_space.items()},
                    **{f'forecaster__{k}': v for k, v in self.y_trans_hyper_search_space.items()},
                    **{f'forecaster__forecaster__{k}': v for k, v in self.y_core_forecaster_hyper_search_space.items()}
                }

        else:
            # Build forecasting pipeline with only y forecaster
            self.forecaster = ForecastingPipeline([('forecaster', transformed_forecaster_y)])
            
    def refit_predict_wrap(self, db_con, forecaster, start_dt, t0, **kwargs):
        """
        Refits the forecaster and generates predictions for the forecast horizon.

        Parameters:
        - db_con (object): Database connection for retrieving historical data.
        - forecaster (dict): A dictionary of individual forecaster models.
        - start_dt (pd.Timestamp): The start date for training data.
        - t0 (pd.Period): The reference timestamp for making predictions.
        - **kwargs: Additional keyword arguments for preprocessing.

        Returns:
        - tuple: (y, X, y_pred), where:
            - y (pd.DataFrame): Target variable data after preprocessing.
            - X (pd.DataFrame): Feature data after preprocessing.
            - y_pred (pd.DataFrame): Forecasted values.
        """

        # Round t0 down to the nearest period based on time series frequency
        t0 = pd.Period(t0).asfreq(freq=self.ts_freq, how='S')

        # Import and preprocess data
        df = self.import_data(db_con=db_con, start_dt=start_dt, end_dt=t0)

        print('Preprocessing data...')
        y, X = self.preprocess_data(
            df=df,
            start_dt=start_dt,
            add_knownX_for_forecast_period=True,
            **kwargs
        )

        # Handling MultiIndex cases
        if isinstance(X.index, pd.MultiIndex):
            ts_ind_name = X.index.levels[-1].name

            # Define training data
            X_train = X[X.index.get_level_values(ts_ind_name) <= t0]
            X_train.index = X_train.index.remove_unused_levels()

            y_train = y[y.index.get_level_values(ts_ind_name) <= t0]
            y_train.index = y_train.index.remove_unused_levels()

            # Define future known X values for forecasting period
            X_fh = X[X.index.get_level_values(ts_ind_name) > t0].fillna(0)

            # Initialize prediction dataframe
            y_pred = pd.DataFrame(index=X_fh.index, columns=['y_pred'], dtype='float')

            # Iterate over forecasters to generate predictions
            for index, model in forecaster.items():
                print(f"Predicting for {index}...")
                
                X_slice = X_train.loc[index]
                y_slice = y_train.loc[index]
                X_fh_slice = X_fh.loc[index]

                # Fit and predict
                model.fit(X=X_slice, y=y_slice, fh=self.forecast_horizon)
                y_pred.loc[index, 'y_pred'] = (
                    model.predict(X=X_fh_slice, fh=self.forecast_horizon)[self.target].values
                    )

        else:
            # Single index case
            X_train = X[X.index <= t0]
            y_train = y[y.index <= t0]
            X_fh = X[X.index > t0]

            # Placeholder for single-index forecasting (implement logic if needed)
            y_pred = None  # Modify this if forecasting logic for single-index is added

        # Drop any NaN values in predictions
        if y_pred is not None:
            y_pred = y_pred.dropna()

        return y, X, y_pred

    def reconcile_hierarchical_forecast(self, df):
        """
        Applies hierarchical reconciliation to the forecasted data.

        Parameters:
        - df (pd.DataFrame): DataFrame containing forecasted values 
          that need hierarchical reconciliation.

        Returns:
        - pd.DataFrame: The reconciled forecasted data.

        Raises:
        - ValueError: If `self.reconciler_method` is not defined.
        """

        if self.reconciler_method is None:
            raise ValueError("Reconciler method is not defined.")

        reconciler = Reconciler(method=self.reconciler_method)
        return reconciler.fit_transform(df)
