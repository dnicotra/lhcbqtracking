import dataclasses as dc
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Rectangle
import numpy as np

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

@dc.dataclass(frozen=True)
class hit:
    hit_id: int
    x: float
    y: float
    z: float
    module_id: int
    track_id: int

    def __getitem__(self, index):
        return (self.x, self.y, self.z)[index]

@dc.dataclass(frozen=True)
class module:
    module_id: int
    z: float
    lx: float
    ly: float
    hits: list[hit]

@dc.dataclass(frozen=True)
class track:
    track_id: int
    track_mc: dict
    hits: list[hit]


@dc.dataclass(frozen=True)
class segment:
    from_hit: hit
    to_hit: hit
    truth: bool = dc.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'truth', (self.from_hit.track_id == self.to_hit.track_id) and (self.from_hit.track_id is not None) and ((self.to_hit.track_id is not None)))

    def to_vect(self):
        return np.array([self.to_hit.x - self.from_hit.x, self.to_hit.y - self.from_hit.y, self.to_hit.z - self.from_hit.z])

    def display(self, ax, equal_axis = True):
        ax.plot((self.from_hit.x, self.to_hit.x),(self.from_hit.y, self.to_hit.y),(self.from_hit.z, self.to_hit.z),lw=.1,c='black')
        
        if equal_axis:
            set_axes_equal(ax)
            ax.set_box_aspect([1,1,1])

@dc.dataclass(frozen=True)
class event:
    modules: list[module]
    tracks: list[track]
    hits: list[hit]

    def display(self, ax: Axes3D, show_tracks = True, show_hits = True, show_modules = True, equal_axis = True):
        if show_hits:
            hit_x, hit_y, hit_z = [], [], []
            for hit in self.hits:
                hit_x.append(hit.x)
                hit_y.append(hit.y)
                hit_z.append(hit.z)
            ax.scatter3D(hit_x, hit_y, hit_z,s=1,c='black')

        if show_modules:
            for module in self.modules:
                p = Rectangle((-module.lx/2, -module.ly/2), module.lx, module.ly,alpha=.2,edgecolor='black')
                ax.add_patch(p)
                art3d.pathpatch_2d_to_3d(p, z=module.z)
        
        if show_tracks:
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            z_lim = ax.get_zlim()
            ts = []
            for track in self.tracks:
                pvx, pvy, pvz = track.track_mc['pv']
                phi = track.track_mc['phi']
                theta = track.track_mc['theta']
                tx1 = max((x_lim[0] - pvx)/(np.sin(theta)*np.cos(phi)),0)
                tx2 = max((x_lim[1] - pvx)/(np.sin(theta)*np.cos(phi)),0)
                ty1 = max((y_lim[0] - pvy)/(np.sin(theta)*np.sin(phi)),0)
                ty2 = max((y_lim[1] - pvy)/(np.sin(theta)*np.sin(phi)),0)
                tz1 = max((z_lim[0] - pvz)/(np.cos(theta)),0)
                tz2 = max((z_lim[1] - pvz)/(np.cos(theta)),0)
                ts.append(min(max(tx1,tx2),max(ty1,ty2), max(tz1,tz2)))

            for track, t in zip(self.tracks, ts):
                pvx, pvy, pvz = track.track_mc['pv']
                phi = track.track_mc['phi']
                theta = track.track_mc['theta']

                ax.plot((pvx,pvx + t*np.sin(theta)*np.cos(phi)),
                (pvy,pvy+ t*np.sin(theta)*np.sin(phi)),
                (pvz, pvz + t*np.cos(theta)))
            
        if equal_axis:
            set_axes_equal(ax)
            ax.set_box_aspect([1,1,1])
        
        ax.set_proj_type('ortho')

 


def vp2q_event(vp_event, lx=None, ly=None):
    import velopix_tracking.event_model.restricted_event_model as vprem
    modules = []
    hits = []

    hit2track = {}

    for partid, particle in enumerate(vp_event.montecarlo['particles']):
        for hitt in particle[-1]:
            hit2track[hitt] = partid

    for mmodule in vp_event.modules:
        module_hits = []
        for hitt in mmodule.hits():
            module_hits.append(hit(hitt.id,hitt.x, hitt.y, hitt.z,mmodule.module_number, hit2track.get(hitt.id)))

        modules.append(module(mmodule.module_number, np.mean(list(mmodule.z)),lx,ly, module_hits))
        hits.extend(module_hits)
        
    q_event = event(modules,None, hits)
    return q_event