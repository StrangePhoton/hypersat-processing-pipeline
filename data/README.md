# Data directory

Nothing in `raw/`, `dem/` or `samples/` is tracked by git, and the pipeline never
downloads anything on its own. Acquisition is an explicit manual step so that runs stay
reproducible and no processing function performs hidden network access.

| Directory | Contents |
| --- | --- |
| `raw/` | Input satellite products, e.g. an unpacked EnMAP L1B product directory |
| `dem/` | Digital elevation models covering the scene footprint with a margin |
| `samples/` | Small rasters for manual experiments (test fixtures are generated in-process instead) |

Quick start:

1. Download an EnMAP **L1B** product from DLR's EOWEB GeoPortal (<https://eoweb.dlr.de>)
   and unpack it into `raw/`. L1B is the useful input: it is in sensor geometry and carries
   RPC coefficients, so there is geometry left to correct. L1C is already orthorectified.
2. Download a DEM covering the scene — Copernicus DEM GLO-30 is the recommended default —
   into `dem/`.
3. Check what you actually received:

   ```bash
   hypersat inspect --input data/raw/<product>
   hypersat validate --input data/raw/<product> --require-rpc --dem data/dem/<dem>.tif
   ```

Full instructions, DEM alternatives and the vertical-datum caveat are in
[../docs/data-sources.md](../docs/data-sources.md).
