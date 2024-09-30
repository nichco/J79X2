import csdl_alpha as csdl
import numpy as np
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
import pickle
import pkg_resources





# custom atmosphere model
class Engine(csdl.CustomExplicitOperation):

    def __init__(self):
        """Max thrust vs. altitude and Mach for 2 J79 engines implemented as a custom explicit operation."""
        super().__init__()
            
        # assign parameters to the class
        path = pkg_resources.resource_filename(__name__, 'data.pkl')
        file = open(path, 'rb')
        dict = pickle.load(file)
        self.func = dict['func']
        self.dx = dict['dx']
        self.dy = dict['dy']


    # def evaluate(self, inputs: csdl.VariableGroup):
    def evaluate(self, altitude: csdl.Variable, mach: csdl.Variable):
        # assign method inputs to input dictionary
        self.declare_input('altitude', altitude)
        self.declare_input('mach', mach)

        # declare output variables
        max_thrust = self.create_output('max_thrust', altitude.shape)

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.max_thrust = max_thrust
        # outputs.temperature = temperature

        return max_thrust
    
    def compute(self, input_vals, output_vals):
        altitude = input_vals['altitude']
        mach = input_vals['mach']

        # output_vals['max_thrust'] = self.func(altitude, mach)
        max_thrust = np.zeros(altitude.shape)
        for i in range(len(altitude)):
            max_thrust = self.func(altitude[i], mach[i])

        output_vals['max_thrust'] = max_thrust

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        altitude = input_vals['altitude']
        mach = input_vals['mach']

        dx = np.zeros(altitude.shape)
        dy = np.zeros(altitude.shape)

        for i in range(len(altitude)):
            dx[i] = self.dx(altitude[i], mach[i])
            dy[i] = self.dy(altitude[i], mach[i])

        derivatives['max_thrust', 'altitude'] = np.diag(dx)
        derivatives['max_thrust', 'mach'] = np.diag(dy)








if __name__ == '__main__':

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    altitude = csdl.Variable(value=30000*1E-3) # kft
    mach = csdl.Variable(value=0.2)

    eng = Engine()
    max_thrust = eng.evaluate(altitude, mach)
    # max_thrust = outputs.max_thrust # klbf

    recorder.stop()

    sim = csdl.experimental.PySimulator(recorder)
    sim.check_totals(ofs=[max_thrust], wrts=[altitude, mach])
    sim.run()


    # print(density.value)
    # print(temperature.value)
    # print(speed_of_sound.value)