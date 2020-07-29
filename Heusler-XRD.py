""" 
Problem:
-- Heusler compounds form a large class of intermetallic materials.
-- They are used for many applications such as hard disc drives and thermoelectrica.
-- The general sum formula is X2YZ with X and Y usually being transition metals. Z is typically given by a main group element.
-- Examples: Cu2MnAl, Fe2MnSi
-- The crystal structure is cubic and has the Strukturberichtnotation L21.
-- Therein the atoms distribute on the four Wyckoff positions 4a, 4b, 4c and 4d.
-- X-ray diffraction allows to determine, which sites are occupied by which atoms.

Goal:
-- The program should calculate the XRD patterns of arbitary Heusler compounds.
-- It should be easy to change the site occupations.
-- The program therefore allows to compare theoretical models of the chemical ordering with experimental data.    
"""

#######################################################################################################

# Import the required libaries

import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
#######################################################################################################


#######################################################################################################

# Import elementspecific data

form_factors = pd.read_csv('Form_Factors.csv', sep=';', header=[0])

#######################################################################################################


#######################################################################################################
# Definition of the material
#######################################################################################################

"""
-- This class creates a material.
-- It describes, which types of atoms are presente, for example Fe2MnSi or Cu2MnAl.
-- Then it calculates the scattering angles of the reflections
-- Finally the atomic form factors are calculated.
"""

class Material():
    
        
    def __init__(self):
                
        # Query of the wavelength             
        
        try:
            self.wavelength = float(input("Which wavelength in Angstrom "))
        except:
            print('Your input is no number. Please correct.')
            self.wavelength = float(input("Which wavelength in Angstrom "))
            
        
        # Query of the number of elements
                    
        try:    
            self.number_elements = int(input('How many elements? '))
        except:
            print('Your input is no number. Please correct.')
            self.number_elements = int(input('How many elements? '))
            
            
        # Query of the types of elements        
            
        self.elemente = ['X1','X2','Y','Z','Y2','Z2']
        self.elemente_x2yz = []
        for i in range(self.number_elements):
            self.elemente_x2yz.append(input("Which {} Element? ".format(self.elemente[i])))
            while self.elemente_x2yz[i] not in form_factors['Element'].values:
                del(self.elemente_x2yz[i])
                print('Your input is no Element')
                self.elemente_x2yz.append(input("Which {} Element? ".format(self.elemente[i])))
                

        # Query of the lattice parameter            
                
        self.a = float(input("Which lattice parameter in Angstrom "))
        while self.wavelength > self.a:
            print('The wavelength must be smaller than the lattice parameter')
            self.a = 0
            self.wavelength = 0
            self.a = float(input("Which lattice parameter in Angstrom "))
            self.wavelength = float(input("Which wavelength in Angstrom "))
            
            
        # The following functions calculate all the required quantities 
            
        self.scattering_coefficients()
        self.reflections()
        self.Var_theta = self.theta()
        self.q()
        self.Var_f = self.f()
        
        
        
    def scattering_coefficients(self):
        
        global form_factors
        coefficients = []
                
        # Takes the element specific scattering constants from the imported table
        
        for i in range(len(self.elemente_x2yz)):
            coefficients.append(form_factors.loc[form_factors['Element'] == self.elemente_x2yz[i]].values)
        coefficients = [i[0] for i in coefficients]
        coefficients = pd.DataFrame(coefficients)
        coefficients = coefficients.loc[:,1:].to_numpy(dtype=float)
        coefficients = np.round(coefficients,decimals=10)
        return coefficients



    def reflections(self):
        
        
        # In order to determine the chemical order of Heusler compounds, it is sufficient to restrict to the following 5 refelctions.
                
        self.hkl=[[1,1,1],[2,0,0],[2,2,0],[1,1,3],[2,2,2]]
               
        self.hkl_string=[]
        for i in range(len(self.hkl)):
            self.hkl_string.append(str(self.hkl[i]))
            self.hkl_string=[j.replace(", ","") for j in self.hkl_string]
            self.hkl_string=[j.replace("[","(") for j in self.hkl_string]
            self.hkl_string=[j.replace("]",")") for j in self.hkl_string]    



    def theta(self):
        
        
        # This function calculates from the (hkl)-values the diffraction angles Theta.
        # Equation: sin(Theta) = (0.5 * wavelength * Wurzel(h^2+k^2+l^2) / lattice parameter) 
                
        hilfsliste = []
        for i in range(len(self.hkl_string)):
            ttheta = np.arcsin((self.wavelength*math.sqrt(self.hkl[i][0]**2+self.hkl[i][1]**2+self.hkl[i][2]**2))/(2*self.a))      
            hilfsliste.append(ttheta)    
        hilfsarray=np.array(hilfsliste)
        return hilfsarray



    def q(self):
        
        
        # This function calculates the scattering vectors q from the diffraction angles Theta.
        # Equation: q = 4 * Pi * sin(Theta) / wavelength
                
        hilfsliste = []
        for i in range(len(self.theta())):
            qq = 4*math.pi*np.sin(self.theta()[i])/self.wavelength
            hilfsliste.append(qq)
        hilfsarray=np.array(hilfsliste)    
        return hilfsarray



    def f(self):
        
        
        # This function calculates the atomic scattering factors.
        # Equation: f(q) = sum over i (a_i exp(-b_i *(q/4Pi)^2) + c)
        # a,b,c are the element specific coefficients and q the scattering vector
                
        hilfsarray = np.zeros(shape=(len(self.hkl_string),len(self.elemente_x2yz)))
        for i in range(len(self.elemente_x2yz)):
            for j in range(len(self.hkl_string)):
                gg = self.scattering_coefficients()[i][8]
                for k in range(0, (len(self.scattering_coefficients()[0])-1),2):
                    gg += self.scattering_coefficients()[i][k]*np.exp(-self.scattering_coefficients()[i][k+1]*((self.q()[j])/(4*np.pi))**2)
                hilfsarray[j][i] = gg
        hilfsarray=np.round(hilfsarray,decimals=10)        
        return hilfsarray
 
#######################################################################################################    
    

#######################################################################################################
# Definition of the chemical ordering
#######################################################################################################

"""
-- Mit dieser Klasse wird eine Atomanordnung erzeugt.
-- Beantwortet die Frage: Auf welchen der vier Wyckoff-Positionen sitzen denn die Atome, die in Material() definiert wurden?
-- Deshalb fungiert das oben erzeugte Material mit seinen Rechenergebnissen als Eingangsvariable.
-- So können zu einem Material leicht mehrere Anordnungen erzeugt werden, um diese dann zu vergleichen.
-- Die Atomanordnung deren resultierende Intensitätsverteilung mit der experimentellen Intensitätsverteilung übereinstimmt,
   liegt dann tatsächlich in dem untersuchten Material vor.
"""

class ChemicalOrder():
    

    def __init__(self,material):
        
        # Takes the created Material as input variable
                
        self.material = material    
        
        # Asks if the is disorder (double occupation).
        # x, y are the disorder parameters.
        
        self.x = 0
        self.y = 0
        disorder = input('Do you need disorder parameters? (yes or no) ')
        
        if disorder != 'yes' and disorder != 'no':
            print('Wrong input. Please repeat.')
            disorder = input('Do you need disorder parameters? (yes or no) ')
        
        if disorder == 'yes':
            How_many = input('How many disorder parameters do you need? (1 or 2) ')
            if How_many == '1':
                try:
                    self.x = np.round(float(input('Value of x? ')),decimals=3)
                except:
                    print('Your input is no number.')
                    self.x = np.round(float(input('Value of x? ')),decimals=3)
            if How_many == '2':
                try:
                    self.x = np.round(float(input('Value of x? ')),decimals=3)
                    self.y = np.round(float(input('Value of y? ')),decimals=3)
                except:
                    print('One of your inputs is no number.')
                    self.x = np.round(float(input('Value of x? ')),decimals=3)
                    self.y = np.round(float(input('Value of y? ')),decimals=3)

         
        # Query of the site occupations.      
        
        self.wyckoff_positions = ['4a','4b','4c','4d']
        self.occupations = {}
        self.occupations_original = {}
        self.occupations_joined = {}
        self.disorder_parameters = {'x': self.x, '(1-x)': (1-self.x), 'y': self.y, '(1-y)': (1-self.y), 'x+y': (self.x+self.y), '1-x-y': (1-self.x-self.y)}
        
        print('\n Which atom sits at which site?\n Examples:\n 0.4 {} + 0.6 {} \n x {} + (1-x) {}'.format(self.material.elemente_x2yz[0],self.material.elemente_x2yz[1],                                                                       
                                                                       self.material.elemente_x2yz[2],self.material.elemente_x2yz[3]))
        
        # The occupations_original-dictionary allows to change x after creating the ChemcialOrder and the update the result.        
        
        for position in self.wyckoff_positions:
            self.occupations[position] = input('What is on {}? '.format(position))
            self.occupations_joined[position] = self.occupations[position]
            self.occupations_original[position] = self.occupations[position]
            self.occupations[position] = self.occupations[position].replace('+','').split()
            for j in range(len(self.occupations[position])):
                try:
                    self.occupations[position][j] = float(self.occupations[position][j])
                except:
                    for key in self.disorder_parameters.keys():
                        if self.occupations[position][j] == key:
                            self.occupations[position][j] = self.disorder_parameters[key]
                else:
                    pass

        
        
        # The following function calculate all the results        
        
        self.position_occupation()
        self.f_position()
        self.structure_factor()
        self.LP_factor()
        self.intensity()
        self.plotting()
        print(self.intensity().iloc[:,1])  
        
            

    def position_occupation(self):
        
        
        # The query of the site occupations is used to create a 2D array.        
        
        dic_wyckoff_positions = {'4a': 0,'4b': 1,'4c': 2,'4d': 3}
        elements = {self.material.elemente_x2yz[0]: 0, self.material.elemente_x2yz[1]: 1, self.material.elemente_x2yz[2]: 2, self.material.elemente_x2yz[3]: 3}
        pos_occupation = np.zeros(shape = (len(self.material.elemente_x2yz),len(dic_wyckoff_positions)))
        
        for i in range(0,len(self.occupations['4a']),2):
            pos_occupation[elements[self.occupations['4a'][i+1]]][dic_wyckoff_positions['4a']] = self.occupations['4a'][i]
        for i in range(0,len(self.occupations['4b']),2):
            pos_occupation[elements[self.occupations['4b'][i+1]]][dic_wyckoff_positions['4b']] = self.occupations['4b'][i]
        for i in range(0,len(self.occupations['4c']),2):
            pos_occupation[elements[self.occupations['4c'][i+1]]][dic_wyckoff_positions['4c']] = self.occupations['4c'][i]
        for i in range(0,len(self.occupations['4d']),2):
            pos_occupation[elements[self.occupations['4d'][i+1]]][dic_wyckoff_positions['4d']] = self.occupations['4d'][i]            
        return pos_occupation        
    
    

    def f_position(self):
        
        
        # Calculates the scattering power of each Wyckoff position        
        
        ff_position = np.dot(self.material.Var_f ,self.position_occupation())
        ff_position = np.round(ff_position, decimals=10)
        return ff_position



    def structure_factor(self):
        
        
        # Calculates the structure factor.
        # The following three rules are valid for Heusler compounds:
        # (1) h,k,l uneven     --> F = 4 * root([f_4a - f_4b]^2 + [f_4c - f_4d]^2 )
        # (2) (h+k+l)/2 = 2n + 1 --> F = 4 * (f_4a + f_4b - f_4c - f_4d)
        # (3) (h+k+l)/2 = 2n     --> F = 4 * (f_4a + f_4b + f_4c + f_4d)
        
        F1 = np.zeros(shape = (len(self.material.hkl_string[::3]),1)) 
        for i in range(len(self.material.hkl_string[::3])):
            F1[i] = 4*math.sqrt((self.f_position()[::3][i][0] - self.f_position()[::3][i][1])**2 + (self.f_position()[::3][i][2] - self.f_position()[::3][i][3])**2)
            
        F2 = np.zeros(shape=(len(self.material.hkl_string[1::3]),1))    
        for i in range(len(self.material.hkl_string[1::3])):
            F2[i] = abs(4*(self.f_position()[1::3][i][0] + self.f_position()[1::3][i][1] - self.f_position()[1::3][i][2] - self.f_position()[1::3][i][3]))
        
        F3 = np.zeros(shape = (len(self.material.hkl_string[2::3]),1))    
        for i in range(len(self.material.hkl_string[2::3])):
            F3[i] = abs(4*(self.f_position()[2::3][i][0] + self.f_position()[2::3][i][1] + self.f_position()[2::3][i][2] + self.f_position()[2::3][i][3]))
        
        F = np.zeros(shape = (len(self.material.hkl_string),1))
        for i in range(len(F1)):
            F[::3][i]= F1[i]
        for i in range(len(F2)):
            F[1::3][i]= F2[i]  
        for i in range(len(F3)):
            F[2::3][i]= F3[i]  
        F = np.round(F,decimals=10)
        return F    



    def LP_factor(self):
       
                
        # Calculates the Lorentz-Polarisations-Faktor LP for Bragg-Brentano geometry.             
        
        LP = []
        for i in range(len(self.material.Var_theta)):
            lp = (1 + (np.cos(2*self.material.Var_theta[i])**2))/(8*np.cos(self.material.Var_theta[i])*(np.sin(self.material.Var_theta[i]))**2)
            LP.append(lp)
        LP = np.array(LP)    
        LP = np.round(LP,decimals=10)
        return LP



    def intensity(self):
        
        
        # The most important step: This function calculates the intensity
        # Equation: Intensity ~ Multiplicity * Lorentz-Polarisations-Factor * Structure_Factor^2
        # Simplified formula since the temperature dependence (Debye-Waller-factor) is not included       
        
        multiplicity = np.array([8,6,12,24,8])
        iintensity = np.zeros(shape = (len(self.material.hkl_string),2))
        for i in range(len(self.material.hkl_string)):
            iintensity[i][0] = self.LP_factor()[i]*multiplicity[i]*(self.structure_factor()[i]**2)
        iintensity = np.round(iintensity, decimals = 0)        
        
        # The second column gives the relative intensity.        
        
        Max_int = np.amax(iintensity[:,0])
        for i in range(len(self.material.hkl_string)):
            iintensity[i][1]=100*iintensity[i][0]/Max_int
        iintensity = np.round(iintensity, decimals = 1)        
        
        iintensity = pd.DataFrame(iintensity, index = self.material.hkl_string, columns = {'Intensity','Rel. Intensity'})
        return iintensity
           
    
        
    def plotting(self):
        
        
        # Ilustrating the results
        # First get degree from the rad values.        
        
        deg_theta = self.material.Var_theta*180/np.pi
        deg_theta = deg_theta.round(decimals = 4)
        deg_2theta = 2*deg_theta
        
        
        # Settings of the plots
        
        
        mpl.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['font.size'] = 20
        
        
        # Create the figure with three axes        
        
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_axes([0, 0, 0.7, 1])
        ax2 = fig.add_axes([0.7, 0.6, 0.3, 0.4])
        ax3 = fig.add_axes([0.85, 0, 0.25, 0.6])
        ax2.axis('off')
        ax3.axis('off')
        ax2.axis('tight')
        ax3.axis('tight')
        
        
        # Creating the bar chart.        
        
        rects = ax1.bar(deg_2theta,self.intensity().iloc[:,1])
        
        
        # The following function labels each reflections.
        
        def autolabel(rects):
            
            for i in range(len(self.material.hkl_string)):                    
                    height = rects[i].get_height()
                    ax1.annotate('{}'.format(self.material.hkl_string[i]),
                                xy=(rects[i].get_x() + rects[i].get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom') 
        

        # Some more settings.                   
            
        ax1.xaxis.set_tick_params(which='major', size=8, width=2, direction='in', top='on')
        ax1.yaxis.set_tick_params(which='major', size=8, width=2, direction='in', top='on')
        ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
        ax1.set_xlim((deg_2theta[0]-3), (deg_2theta[-1]+3))
        ax1.set_ylim(0, 115)
        ax1.set_xlabel(r'2$\theta$ ($^{\circ}$)')
        ax1.set_ylabel(r'Relative Intensity (%)')
        plt.rcParams['font.size'] = 16
        autolabel(rects)
        
        
        # Giving a text box with the chosen site occupations.
        
        plt.rcParams['font.size'] = 18
        try:
            for value in self.occupations_joined:
                    self.occupations_joined[value] = self.occupations_joined[value].replace('x ',(str(self.x)+' '))
                    self.occupations_joined[value] = self.occupations_joined[value].replace('(1-x)',str(round((1-self.x),2)))
                    self.occupations_joined[value] = self.occupations_joined[value].replace('y ',str(self.y)+' ')
                    self.occupations_joined[value] = self.occupations_joined[value].replace('(1-y)',str(round((1-self.y),2)))
        except:
            pass
        ax2.text(0.2,0.1,' 4a: {} \n 4b: {} \n 4c: {} \n 4d: {}'.format(self.occupations_joined['4a'],
                                                                         self.occupations_joined['4b'],
                                                                         self.occupations_joined['4c'],
                                                                         self.occupations_joined['4d']))
        
        
        # Gives a table with the relative intensities.        
        
        table = ax3.table(cellText = np.reshape(self.intensity().iloc[:,1].to_numpy(),(len(self.material.hkl_string),1)),
                 rowLabels = self.material.hkl_string,
                 colLabels = ['Rel. Intensity (%)'],
                 bbox = [0,0,1,0.9],
                 cellLoc = 'center',
                 colLoc = 'center')
        table.scale(1,2)
        
        
        plt.show()
        
        
        
    def update(self):
        
        
        # Allows to update for example the disorder parameters x and y.
        
        
        self.disorder_parameters = {'x': self.x, '(1-x)': (1-self.x), 'y': self.y, '(1-y)': (1-self.y), 'x+y': (self.x+self.y), '1-x-y': (1-self.x-self.y)}    
            
        for position in self.wyckoff_positions:
            self.occupations[position] = self.occupations_original[position].replace('+','').split()
            self.occupations_joined[position] = self.occupations_original[position]
            for j in range(len(self.occupations[position])):
                try:
                    self.occupations[position][j] = float(self.occupations[position][j])
                except:
                    for key in self.disorder_parameters.keys():
                        if self.occupations[position][j] == key:
                            self.occupations[position][j] = self.disorder_parameters[key]
                else:
                    pass
          
            
        self.position_occupation()
        self.f_position()
        self.structure_factor()
        self.LP_factor()
        self.intensity()
        self.plotting()
        print(self.intensity().iloc[:,1])  

#######################################################################################################
        

print('\
      (1) Create a material with the Material()-Class \n\
          This could for example be Fe2MnSi = Material() \n\
      (2) Define your crystal structure with the ChemicalOrder()-Class \n\
          It needs a material as parameter.\n\
          This could for example be L21 = ChemicalOrder(Fe2MnSi)\n\
      (3) Repeat step 2 until you find the correct atomic arrangement.')


#######################################################################################################
      
"""
Example 1:
       
        Fe2MnSi = Material()
        wavelength = 1.54
        number of Elements = 4
        X1 = Fe
        X2 = Fe
        Y = Mn
        Z = Si
        Lattice parameter a = 5.6
        
        L21 = ChemicalOrder(Fe2MnSi)
        Disorder? no
        4a: 1 Si
        4b: 1 Mn
        4c: 1 Fe
        4d: 1 Fe

Example 2:
       
        IrMnGa = Material()
        wavelength = 1.54
        number of Elements = 4
        X1 = Ir
        X2 = Mn
        Y = Ga
        Z = Vac
        Lattice parameter a = 6
        
        C1bII = ChemicalOrder(IrMnGa)
        Disorder? yes
	x = 0.5
        4a: x Ga + (1-x) Ir
        4b: x Ir + (1-x) Ga
        4c: 1 Mn
        4d: 1 Vac
        
        
"""     