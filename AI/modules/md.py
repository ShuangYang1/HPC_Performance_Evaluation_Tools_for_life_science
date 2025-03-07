from sponge import Sponge
from sponge import ForceField
from sponge.optimizer import SteepestDescent
from sponge.callback import WriteH5MD, RunInfo
from sponge import set_global_units
from sponge import Protein
from sponge import UpdaterMD
from sponge.function import VelocityGenerator
from mindspore import context
import sys

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
set_global_units('nm', 'kj/mol')
pdb_name = sys.argv[1]
system = Protein(pdb=pdb_name, rebuild_hydrogen=True)
energy = ForceField(system, 'AMBER.FF14SB')
min_opt = SteepestDescent(system.trainable_params(), 1e-7)
md = Sponge(system, energy, min_opt)
run_info = RunInfo(10)
md.run(500, callbacks=[run_info])
vgen = VelocityGenerator(300)
velocity = vgen(system.shape, system.atom_mass)
opt = UpdaterMD(system=system,
                time_step=1e-3,
                velocity=velocity,
                integrator='velocity_verlet',
                temperature=300,
                thermostat='langevin',)
md = Sponge(system, energy, min_opt)
cb_h5md = WriteH5MD(system, sys.argv[2], save_freq=10, write_velocity=True, write_force=True)
md.change_optimizer(opt)
md.run(2000, callbacks=[run_info, cb_h5md])