# -*- coding: utf-8 -*-
"""Collection of helper scripts, particulary for dealing with spatial data."""

import random
from collections import namedtuple
from typing import Optional, Union

import numpy.typing as npt
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file
from shapely.geometry import Point, Polygon
from sklearn.cluster import AgglomerativeClustering

BufferPoint = namedtuple("BufferPoint", "point, distance")


def verify_float_range(x: float) -> float:
    """Verify that `x` is within the range [0,1], else return a valid random number."""
    return x if isinstance(x, float) and 0.0 <= x <= 1.0 else random.random()  # noqa: PLR2004


def cluster_shapes(
    geodf: GeoDataFrame,
    num_clusters: int = 2,
    distance: float = None,
    check_crs: bool = True,
) -> npt.NDArray:
    """Make groups for all shapes within a defined distance.

    For a shape to be excluded from a group, it must be greater than the defined
    distance from *all* shapes in the group. Distances are calculated using shape
    centroids.

    Args:
        geodf: A geopandas data.frame of polygons. Should be a projected CRS where the
            unit is in meters.
        num_clusters: The number of clusters to make. will be set to *None* if
            *distance* is not *None*.
        distance: Maximum distance between elements. In metres.
        check_crs: Confirm that the CRS of the geopandas dataframe is projected. This
            function should not be run with lat/lon coordinates.

    Returns:
        Array of numeric labels assigned to each row in geodf.
    """
    if check_crs:
        assert (
            geodf.crs.is_projected
        ), "geodf should be a projected crs with meters as the unit"

    centers = [p.centroid for p in geodf.geometry]
    centers_xy = [[c.x, c.y] for c in centers]

    if distance is not None:
        num_clusters = None

    cluster = AgglomerativeClustering(
        n_clusters=num_clusters,
        linkage="ward",
        metric="euclidean",
        distance_threshold=distance,
    )
    cluster.fit(centers_xy)

    return cluster.labels_


def get_geopoint(lat: float, lon: float, from_epsg: int, to_epsg: int) -> GeoSeries:
    """Generate and return a geopandas point from the lat and lon coordinates.

    Args:
        lat: The latitude of the point.
        lon: The longitude of the point.
        from_epsg: The espg of the longitude, latitude.
        to_epsg: The desired epsg to project the point to.

    Returns:
        the corresponding point as a GeoSeries
    """
    return GeoSeries(points_from_xy([lon], [lat], crs=f"EPSG:{from_epsg}")).to_crs(
        epsg=to_epsg,
    )


PointBuffer = namedtuple("PointBuffer", "point, distance")
PointBuffer.__doc__ = """\
    A buffer around a geographic point.

    point - the point around which the buffer is set
    distance - the distance of the buffer around the point"""


def read_shapefile(
    shapefile_path: str,
    /,
    epsg: Optional[int] = None,
    mask: Optional[Union[GeoSeries, GeoDataFrame]] = None,
    features_to_drop: Optional[list[str]] = None,
    cols_to_drop: Optional[list[str]] = None,
) -> GeoDataFrame:
    """Read a shapefile, apply mask and filters are specified and return a geodataframe.

    Args:
        shapefile_path: The path where the shapefile is located.
        epsg: The epsg for the target crs of the shapefile.
        mask: Filter features that intersect with the mask.
        features_to_drop: Any features to drop from the shapefile before returning.
        cols_to_drop: Any columns to drop from the shapefile before returning.

    Returns:
        the dataframe representing the shapefile
    """
    gdf = read_file(shapefile_path, mask)

    if features_to_drop:
        gdf = gdf[
            ~gdf.Secondary.str.contains(
                "|".join(features_to_drop),
                case=False,
                regex=True,
            )
        ]
    if cols_to_drop:
        gdf = gdf.drop(columns=cols_to_drop)
    if epsg:
        gdf = gdf.to_crs(epsg=epsg)

    return gdf


def random_locations(n: int, polygon: Polygon) -> list[Point]:
    """Get random locations within a polygon.

    Args:
        n: The number of locations to get.
        polygon: The polygon witthin which to get the locations.

    Returns:
        A list of locations within the polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds

    points = []
    while len(points) < n:
        pt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pt):
            points.append(pt)

    return points
