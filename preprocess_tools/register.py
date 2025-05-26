import numpy as np
import cv2

def frontwall_orientation(volume, compare_axis, reference_axis='Z'):

    """
    This function is used to register the front wall orientation
    by estimating its inclination angle in a 2D plane.

    The orientation is computed in the plane formed by a fixed reference axis
    (default is 'Z') and a user-defined comparison axis ('X' or 'Y').

    Parameters
    ----------
    volume : numpy.ndarray
        3D volume with shape (Z, Y, X).
    reference_axis : str
        The axis used as reference for inclination estimation. Default is 'Z'.
        Only 'Z' is currently supported.
    compare_axis : str
        The secondary axis used to define the 2D plane with the reference axis.
        Must be either 'X' or 'Y'.

    Returns
    -------
    angle_deg : float
        Estimated inclination angle (in degrees) of the front wall in the plane
        defined by (reference_axis, compare_axis).
    """

    # Mapeo de nombres a índices
    axis_map = {'Z': 0, 'Y': 1, 'X': 2}

    """La idea del mapeo esta bien pero es mejor que los ejes se traten desde el principio como 0,1,2.
        Es menos intuitivo de cara a nosotros dos que sabemos que Z es 0, Y es 1 y X es 2, pero es más fácil de entender para el código.
        Además, así no hay que hacer el mapeo de nombres a índices.

        Si alguien quiere usar esta funcion con un volumen que tiene el sistema de ejes cambaido seria un lio para entenderlo si no estan en 0,1,2.

        Asique que en la funcion entren los ejes como 0,1,2 desde el principio aunque como comentario quieras explicar cual es cual para ti.
    """
    
    reference_axis = reference_axis.upper()
    compare_axis = compare_axis.upper()

    if compare_axis not in axis_map:
        raise ValueError("compare_axis must be 'X' or 'Y'.")
    
    if compare_axis == reference_axis:
        raise ValueError("compare_axis must be different from reference_axis.")


    # Convertir a índices
    ref_axis_idx = axis_map[reference_axis]
    comp_axis_idx = axis_map[compare_axis]

    # Eje restante para colapsar y obtener el B-scan en el plano (Z vs X) o (Z vs Y)
    collapse_axis = list({0, 1, 2} - {ref_axis_idx, comp_axis_idx})[0]


    def compute_angle(volume, collapse_axis):
        b_scan = np.max(volume, collapse_axis)
        mask = (b_scan > 100).astype(np.uint8) * 255
        bscan_uint8 = b_scan.astype(np.uint8)
        bscan_mask = cv2.bitwise_and(bscan_uint8, bscan_uint8, mask=mask)
        ys, xs = np.where(bscan_mask > 0)
        if len(xs) < 2:
            return None
        A = np.vstack([xs, np.ones_like(xs)]).T
        m, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
        
        return np.degrees(np.arctan(m))

    angle = compute_angle(volume, collapse_axis)

    return angle

""" Cuando acabes la funcion, dile a copilot o a chatgpt lo siguiente: Please properly comment and document this code."""