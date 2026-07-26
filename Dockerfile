# Reference runtime for the HyperSat pipeline.
#
# The image is based on python:slim rather than on an osgeo/gdal image because the
# rasterio wheels bundle their own libgdal, PROJ database and PROJ grids. That keeps the
# container reproducible and identical to a plain `pip install` on a developer machine.
#
# If you need the `osgeo.gdal` Python bindings (optional `gdal` extra), switch the base
# image to `ghcr.io/osgeo/gdal:ubuntu-small-<version>` and install without that extra's
# pip build, since those images already provide matching bindings. See
# docs/data-sources.md for the trade-off.
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # GDAL/PROJ runtime tuning: 512 MB block cache, and fail loudly instead of
    # silently returning empty data when a remote/relative path is wrong.
    GDAL_CACHEMAX=512 \
    GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
    HYPERSAT_LOG_FORMAT=json

# libgomp1 is required by the OpenMP-enabled NumPy/OpenCV wheels.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what the build backend needs first, so dependency layers stay cached.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY configs ./configs

# Mount points for inputs and results; nothing large is baked into the image.
RUN mkdir -p /app/data/raw /app/data/dem /app/outputs \
    && useradd --create-home --uid 10001 hypersat \
    && chown -R hypersat:hypersat /app
USER hypersat

ENTRYPOINT ["hypersat"]
CMD ["--help"]
