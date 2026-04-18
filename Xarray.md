A Python library for working with labeled multidimensional arrays.
It extends [[NumPy]] by adding named dimensions and coordinate labels, making it the primary tool for scientific array data (climate, oceanography, atmospheric science, and increasingly Earth observation).

Two core ata structures:

DataArray: A single labeled N-dimensional array
```python
  import xarray as xr
  import numpy as np                                                                                
  # A temperature array with named dimensions and coordinates             
  temp = xr.DataArray(
      data=np.random.rand(365, 180, 360),                                 
      dims=["time", "lat", "lon"],                                        
      coords={
          "time": pd.date_range("2023-01-01", periods=365),               
          "lat": np.linspace(-90, 90, 180),
          "lon": np.linspace(-180, 180, 360)                              
      },                                                                  
      attrs={"units": "K", "long_name": "Air Temperature"}
  )                                                                       
```

Dataset: A collection of DataArrays sharing the same dimension; equivalent to a [[Network Common Data Form|NetCDF]] file in memory.
```python
ds = xr.Dataset({                                                        
  "temperature": temp,
  "humidity": xr.DataArray(...),
  "wind_u": xr.DataArray(...)                                         
}) 
```

