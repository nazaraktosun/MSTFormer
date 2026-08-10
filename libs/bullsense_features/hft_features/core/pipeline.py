import pathlib,pickle,joblib
from hft_features.core.base import _REGISTRY
import polars as pl


class Pipeline:
    def __init__(self, feature_specs : list[dict], cache_dir : str = None):
        """
        Initializes the pipeline by creating feature instances from specifications.

        Args:
            feature_specs (list[dict]): A list of feature configurations,
                e.g., [{"name": "rolling_std", "params": {"window": 20}}].
            cache_dir (str, optional): The directory to store and load cached results.
                                       If None, caching is disabled.
        """
        #Instantiate each feature object using the _REGISTERY
        self.steps = [
            _REGISTRY[s["name"]](**s.get("params", {}))
            for s in feature_specs
        ]

        #set up the cache directory path if provided
        if cache_dir:
            self.cache = pathlib.Path(cache_dir)
            self.cache.mkdir(parents=True, exist_ok=True) #ensure directory exists
        else:
            self.cache = None

    def fit_transform(self, df : pl.DataFrame) -> pl.DataFrame:
        """
         Applies each feature step in order to the DataFrame.

         For each step, it checks for a cached result. If found, it loads it.
         If not, it computes the feature and saves the result to the cache.

         Args:
             df (pd.DataFrame): The input DataFrame.

         Returns:
             pd.DataFrame: The transformed DataFrame with all new feature columns.
         """
        print(f"Starting pipeline with {len(self.steps)} steps...")
        for i, step in enumerate(self.steps):
            sig = step.signature()
            print(f"Step {i + 1}/{len(self.steps)}: {step.name}({step.params}) -> signature: {sig[:8]}...")

            # Define expected cache file (only when caching is enabled)
            cache_file = self.cache / f"{sig}.parquet" if self.cache else None

            # Use cache when available; otherwise compute the feature
            if self.cache and cache_file.exists():
                print("-> Found cache. Loading from disk.")
                df = pl.read_parquet(cache_file)
            else:
                # Compute the transformation
                print("-> Computing feature (no cache).")
                df_out = step(df)

                if self.cache:
                    print(f"-> Saving cache to {cache_file}.")
                    df_out.write_parquet(cache_file)

                df = df_out
        print("Pipeline finished")
        return df


