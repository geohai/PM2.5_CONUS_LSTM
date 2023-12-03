# Daily 1-KM resolution PM2.5 for Contiguous US from 2015 to 2021

The Repo holds the script and models for generating 
**Daily 1-KM resolution PM2.5** for 
contiguous US from 2015-08-26 to 2021-12-31

## Description

* Data

| Data Source                | Variable                                                                                    | Description                                                                                                                                                                             | Temporal Resolution | Spatial Resolution  |
|----------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|---------------------|
| MODIS MCD19A2.061          | AOD 047 and AOD 055                                                                         | MODIS Terra and Aqua combined Multi-angle Implementation of Atmospheric Correction (MAIAC)  Land Aerosol Optical Depth (AOD) gridded Level 2 product produced daily at 1 km resolution. | Daily               | 1-km                |
| MODIS MCD43A4_006_NDVI     | NDVI                                                                                        | NDVI generated from the MODIS/006/MCD43A4 surface reflectance composites.                                                                                                               | 16-Day              | 0.5-km              |
| Daymet                     | Day Length, Pricipitation, Shortwave Radiation,  Max and Min Temperature, and Vapor Pressue | Daily Surface Weather and Climatological Summaries                                                                                                                                      | Daily               | 1-km                |
| gridMet                    | Wind direction and velocity                                                                 | A dataset of daily high-spatial resolution (~4-km, 1/24th degree) surface meteorological data covering  the contiguous US from 1979-yesterday.                                          | Daily               | 1/24th degree ~4-km |
| GMTED                      | DEM                                                                                         | The Global Multi-resolution Terrain Elevation Data 2010 (GMTED2010).                                                                                                                    | Static (2010)       | 1-km                |
| NOAA Hazard Mapping System | Wildfire Smoke Density                                                                      | NOAA's Hazard Mapping System (HMS) Smoke Product                                                                                                                                        | Daily               | 1-km                |

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

[@Zhongying Wang](Zhongying.Wang@colorado.edu)

## Version History

* 2.0
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 1.0
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.