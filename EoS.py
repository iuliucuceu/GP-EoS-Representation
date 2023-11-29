import torch
import numpy as np

C_SI = 299792458.0  # m/s
C_CGS = C_SI * 100.
G_SI = 6.6743e-11  # m^3 kg^-1 s^-2
MSUN_SI = 1.9884099021470415e+30  # Kg
# Stores conversions from geometerized to cgs or si unit systems
conversion_dict = {'pressure': {'cgs': C_SI ** 4. / G_SI * 10., 'si': C_SI ** 4. / G_SI, 'geom': 1.},
                   'energy_density': {'cgs': C_SI ** 4. / G_SI * 10., 'si': C_SI ** 4. / G_SI, 'geom': 1.},
                   'density': {'cgs': C_SI ** 2. / G_SI / 1000., 'si': C_SI ** 2. / G_SI, 'geom': 1.},
                   'pseudo_enthalpy': {'dimensionless': 1.},
                   'mass': {'g': C_SI ** 2. / G_SI * 1000, 'kg': C_SI ** 2. / G_SI, 'geom': 1.,
                            'm_sol': C_SI ** 2. / G_SI / MSUN_SI},
                   'radius': {'cm': 100., 'm': 1., 'km': .001},
                   'tidal_deformability': {'geom': 1.}}

class log_EoS:
    # All numerical data is stored in SI as log10() and 1/C_SI^2
    def __init__(self, pressure, energy_density, phi=None, unit_system={'p': 'si', 'mu': 'si', 'rho': 'si'}):
        self.pressure = torch.tensor(pressure - 2*np.log10(C_CGS), dtype=torch.float64)
        self.energy_density = torch.tensor(energy_density - 2*np.log10(C_CGS), dtype=torch.float64)
        self.phi = torch.tensor(phi, dtype=torch.float64) if phi is not None else None
        self.unit_system = unit_system
        # self.calculate_density()
        self.calculate_phi()
        self.mask = self.pressure > 10
        
    def calculate_density(self): #FIXME
        # Compute the density (assuming energy density = rest mass energy density + internal energy)
        # and that rest mass energy density dominates (approximation)
        # Units will always be in cgs, so convert energy density (temporary) to cgs first
        if self.unit_system['mu'] == 'si':
            e_d = self.energy_density + 2*np.log10(C_CGS) + np.log10(conversion_dict['energy_density']['cgs'])
            self.density = e_d - np.log10(C_CGS ** 2.)
        elif self.unit_system['mu'] == 'cgs':
            self.density = self.energy_density - np.log10(C_CGS ** 2.)
        self.unit_system['rho'] = 'cgs'

    def calculate_phi(self):
        pass

class EoS_family:
    def __init__(self):
        self.eos_list = []

    def add_eos(self, pressure, energy_density, phi=None, unit_system={'p': 'si', 'mu': 'si', 'rho': 'si'}):
        # Add a new EoS to the family and convert from geom to si
        pressure, energy_density = self.convert_units(pressure, energy_density)
        eos = log_EoS(pressure, energy_density, phi, unit_system={'p': 'si', 'mu': 'si', 'rho': 'si'})
        self.eos_list.append(eos)

    def check_file_format_and_units(self, file_path):
        # Check if the file is tabulated or not
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            if not first_line:
                raise ValueError("The file is empty or does not contain the required unit information in the first line.")

            try:
                unit_parts = first_line.split(' ')
                if len(unit_parts) != 4 or unit_parts[0] != 'p' or unit_parts[2] != 'mu' or '[' not in unit_parts[1] or '[' not in unit_parts[3]:
                    raise ValueError("The first line format is incorrect. Expected format: 'p [units] mu [units]'")
                pressure_unit = unit_parts[1][unit_parts[1].find('[')+1 : unit_parts[1].find(']')]
                energy_density_unit = unit_parts[3][unit_parts[3].find('[')+1 : unit_parts[3].find(']')]
                return pressure_unit, energy_density_unit
            except IndexError:
                raise ValueError("The first line of the file does not contain the expected unit information.")

    def read_tabulated_eos(self, file_path):
        # All data is saved as log10
        pressure_unit, energy_density_unit = self.check_file_format_and_units(file_path)
        with open(file_path, 'r') as file:
            # Skip the first line
            file.readline()
            pressure = []
            energy_density = []
            for line in file:
                p, mu = line.split()
                pressure.append(float(p))
                energy_density.append(float(mu))
            self.add_eos(np.log10(pressure), np.log10(energy_density), unit_system={'p': pressure_unit, 'mu': energy_density_unit})

    def convert_units(self, pressure, energy_density):
        # geom -> CGS
        target_unit_system = 'cgs'
        # Convert pressure and energy density
        conversion_factors = {
            'pressure': conversion_dict['pressure'][target_unit_system],
            'energy_density': conversion_dict['energy_density'][target_unit_system],
            'density': conversion_dict['density'][target_unit_system]
        }

        pressure = pressure + np.log10(conversion_factors['pressure'])
        energy_density = energy_density + np.log10(conversion_factors['energy_density'])
        return pressure, energy_density

    def EoS_fitting(self, eos_index, n=3):
        # Fit a polynomial to the log(p) vs log(μ) data above log(p) = 10
        eos = self.eos_list[eos_index]
        coef = np.polyfit(eos.pressure[eos.mask], eos.energy_density[eos.mask], n)
        return coef

    def get_residuals(self, eos_index, n=3):
        # Get the residuals of the polynomial fit
        eos = self.eos_list[eos_index]
        coef = self.EoS_fitting(eos_index, n)
        residuals = (10.**eos.energy_density[eos.mask] - 10.**(np.polyval(coef, eos.pressure[eos.mask]))) / 10.**np.polyval(coef, eos.pressure[eos.mask])
        return residuals

    def plot_residuals(self, eos_index, n=3):
        eos = self.eos_list[eos_index]
        residuals = self.get_residuals(eos_index, n)
        plt.plot(eos.pressure[eos.mask], residuals, 'x', markersize=1, color='red')
        plt.xlabel('log(p)')
        plt.ylabel('Residuals')
        plt.show()

    def plot_eos(self, eos_index, n=3):
        eos = self.eos_list[eos_index]
        coef = self.EoS_fitting(eos_index, n)
        plt.plot(eos.pressure[eos.mask], eos.energy_density[eos.mask], '.', label='EoS', color='red', markersize=10)
        plt.plot(eos.pressure[eos.mask], np.polyval(coef, eos.pressure[eos.mask]) , label='Fit')
        plt.xlabel('log(p)')
        plt.ylabel('log(μ)')
        plt.legend()
        plt.show()
