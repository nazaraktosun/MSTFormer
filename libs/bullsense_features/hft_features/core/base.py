import inspect, functools,hashlib, json
import polars as pl

_REGISTRY = {}
# This is a simple python dict that wll map feature names feature classes
#It lets pipeline look up any feature by name at runtime
#Kind of like a central phone book


class FeatureMeta(type):
    """Allow decorated features to work as pipeline steps and direct helpers."""

    def __call__(cls, *args, **kwargs):
        if args and isinstance(args[0], pl.DataFrame):
            df = args[0]
            if len(args) > 1:
                raise TypeError(
                    f"{cls.__name__} direct calls accept one DataFrame positional argument"
                )
            return super().__call__(**kwargs)(df)
        return super().__call__(*args, **kwargs)


class Feature(metaclass=FeatureMeta):
    """
    Pure transformer: df_in -> df_out with new cols only
    """
    name: str
    params:dict
    deps: set[str]

    def __init_subclass__(cls, **kwargs):
        """
        Runs everytime we create a new subclass of Feature not
        when you create instance of Feature!!!!
        automatically adds MyFeature to _REGISTRY like auto
        registration
        """
        if not hasattr(cls,"name"):
            cls.name = cls.__name__ #holds name of class as string
        _REGISTRY[cls.name] = cls #auto registration step

    def __init__(self, **params):
        """
        Standard constructor for Feature
        It stores the specific settings (hyperparameters) for one particular use of a feature.
        In this example, self.params would become {'window': 20}.
        It also figures out which other features this one depends on (deps).
        ** collect any keyword arguments and puts them in params dictionary
        """
        self.params = params
        self.deps = set(getattr(self,"requires",[])) #sets ups features dependencies


    def __call__(self, df:pl.DataFrame) -> pl.DataFrame:
        """
        It defines the core contract: every feature must be a "pure transformer."
        It takes a DataFrame, adds its new column(s), and returns the modified DataFrame.
        Raising NotImplementedError forces you to implement this for every feature.
        """
        raise NotImplementedError

    def signature(self) -> str:
        """
        Creates a unique id (hash) for this feature and its specific parameters.
        Builds a reproducible JSON of {name,params}, hashes it, and returns a short unique key.
        Used for cache-lookup: if you’ve already run this exact feature
        with these exact parameters, you can load its results instead of recomputing.
        """
        cfg = dict(name = self.name, params = self.params)
        return hashlib.md5(
            json.dumps(cfg, sort_keys=True).encode("utf-8")
        ).hexdigest()

def feature(fn = None, *, deps = ()):
    """
    Decorator to turn a bare function into a Feature subclass instantly.
    * means all arguments later on must specify keyword name you must use
    deps = ... not just pass a tuple
    """
    if fn is None:
        return functools.partial(feature, deps = deps)

    class _InlineFeature(Feature):
        name = fn.__name__
        requires = deps
        __doc__ = fn.__doc__
        def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
            return fn(df, **self.params)

    return _InlineFeature


