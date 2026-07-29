"""
Database models and helpers for storing fusion results in SQLite.
"""

# =============================================================================
# FILENAME:       results_db.py
# DESCRIPTION:    Ce fichier est le fichier charger de la création de l base de données SQL.
#  
# REPOSITORY:     https://github.com/comsee-research/VISWIR.git
#
# AUTHOR:         [Riffard Alexandre]
# EMAIL:          [alexandre.riffard@uca.fr]
# CREATED:        [09-05-2025]
# LAST UPDATED:   [09-05-2025]
# VERSION:        1.0
#
# LICENSE:        GNU LESSER GENERAL PUBLIC LICENSE (voir LICENSE dans le dépôt)
#
# USAGE:          - Appeler ce fichier dans "batch_processing.py"
#
# DEPENDENCIES:   - sqlalchemy
#
# NOTES:
#   - ...
#
# CHANGELOG:
#   - [09-05-2025]: Création initiale du fichier.
#   - [..-..-....]: ...
#
# =============================================================================

from sqlalchemy import create_engine, Column, String, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json

Base = declarative_base()

class FusionResult(Base):
    """
    SQLAlchemy ORM model for storing VIS–SWIR fusion results.

    Attributes
    ----------
    id : int
        Primary key identifier.
    visible_img : str
        Path to the visible image used in the fusion.
    swir_img : str
        Path to the SWIR image used in the fusion.
    ref_img : str
        Path to the reference image, if available.
    grd_tr : str
        Path to the ground truth image, if available.
    alpha : float
        Alpha parameter used in the fusion process.
    beta : float
        Beta parameter used in the fusion process.
    level : float
        Fusion level parameter.
    gamma : float
        Gamma correction value.
    metrics_f : str
        Fusion metrics stored as a JSON string.
    metrics_v : str
        Visible image metrics stored as a JSON string.
    metrics_s : str
        SWIR image metrics stored as a JSON string.
    error : str, optional
        Error message if the fusion process failed.
    """

    __tablename__ = 'fusion_results'

    id = Column(Integer, primary_key=True)
    visible_img = Column(String)
    swir_img = Column(String)
    ref_img = Column(String)
    grd_tr = Column(String)
    alpha = Column(Float)
    beta = Column(Float)
    level = Column(Float)
    gamma = Column(Float)
    metrics_f = Column(String)  # Stocke les métriques sous forme de JSON
    metrics_v = Column(String)  # Stocke les métriques sous forme de JSON
    metrics_s = Column(String)  # Stocke les métriques sous forme de JSON
    error = Column(String, nullable=True)

def get_session(db_path="results.db"):
    """
    Create a new SQLAlchemy session connected to the results database.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite database file (default is "results.db").

    Returns
    -------
    Session
        A SQLAlchemy session object bound to the database.
    """

    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def save_result_to_db(session, visible_img, swir_img, ref_img, grd_tr, alpha, beta, level, gamma, metrics_dict_f, metrics_dict_v, metrics_dict_s, error=None):
    """
    Save a new fusion result entry into the database.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    visible_img : str
        Path to the visible image.
    swir_img : str
        Path to the SWIR image.
    ref_img : str
        Path to the reference image.
    grd_tr : str
        Path to the ground truth image.
    alpha : float
        Alpha parameter used in the fusion process.
    beta : float
        Beta parameter used in the fusion process.
    level : float
        Fusion level parameter.
    gamma : float
        Gamma correction value.
    metrics_dict_f : dict or str
        Fusion metrics (dictionary or JSON string).
    metrics_dict_v : dict or str
        Visible image metrics (dictionary or JSON string).
    metrics_dict_s : dict or str
        SWIR image metrics (dictionary or JSON string).
    error : str, optional
        Error message if the fusion process failed.

    Notes
    -----
    - Metrics dictionaries are automatically converted to JSON strings.
    - The result is committed immediately to the database.
    """
    
    def ensure_json(data):
        """
        Convert data to JSON string if it is not already a string.
        """
        if data is None:
            return None
        if isinstance(data, str):
            return data  # déjà une chaîne JSON
        return json.dumps(data)
    
    result = FusionResult(
        visible_img=visible_img,
        swir_img=swir_img,
        ref_img=ref_img,
        grd_tr=grd_tr,
        alpha=alpha,
        beta=beta,
        level=level,
        gamma=gamma,
        # metrics_f=json.dumps(metrics_dict_f),  # Convertir en JSON
        # metrics_v=json.dumps(metrics_dict_v),  # Convertir en JSON
        # metrics_s=json.dumps(metrics_dict_s),  # Convertir en JSON
        metrics_f=ensure_json(metrics_dict_f),
        metrics_v=ensure_json(metrics_dict_v),
        metrics_s=ensure_json(metrics_dict_s),
        error=error
    )
    session.add(result)
    session.commit()
