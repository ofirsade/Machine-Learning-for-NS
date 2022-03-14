import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from typing import List
import pyrfume
import pubchempy as pcp

# specify the input path for the dataset
input_path_data = r"C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\12868_2016_287_MOESM1_ESM.xlsx"

# loading the dataset
df2 = pd.read_excel(input_path_data, header=2)

# keeping only index information and the 20 odor characters
df2 = df2[["CID", "Odor","Odor dilution","EDIBLE ","BAKERY ","SWEET ","FRUIT ", "FISH", "GARLIC ","SPICES ", "COLD", "SOUR","BURNT ", "ACID ", "WARM ", "MUSKY ", "SWEATY ","AMMONIA/URINOUS","DECAYED","WOOD ","GRASS ","FLOWER ", "CHEMICAL "]]

# removing spaces from columns names
df2.columns = df2.columns.str.replace(' ', '')

# changing 'Odordilution' to 'Odor_dilution
df2.rename(columns={'Odordilution': 'Odor_dilution'}, inplace=True)

# replacing na with 0 for future mean calculation
df2.fillna(0,inplace=True)

# replacing wrong CID's that contain "-"
cid = df2['CID']
odor = df2['Odor']
for i in range(len(cid)):
    if "-" in str(cid[i]):
        s = pcp.get_compounds(odor[i], 'name')
        cid[i] = s[0].cid
df2["CID"] = cid

# grouping the dataset by odor and dilution. each row will contain the mean of every odor charcter
df2 = df2.groupby(['CID', 'Odor_dilution'], sort=False).mean()

# finding the 3 max means for every combination of odor and dilution
c = ['1st Max','2nd Max','3rd Max']
df2 = df2.apply(lambda x: pd.Series(x.nlargest(3).index, index=c), axis=1).reset_index()

# 1) adding "SMILES" to the df using, "CID". 2) for each row, checks if "SOUR" is in max 3 means. 
y = []
smiles=[]
cid = df2['CID']
for i in range(len(df2)):
    row = df2.iloc[i].values
    y.append(int("SOUR" in row))
    s = pcp.Compound.from_cid(str(cid[i]))
    smile = s.isomeric_smiles
    smiles.append(smile)

# adding "SMILES" to the df
df2["SMILES"] = smiles
df2 = df2[["CID", "Odor_dilution", "SMILES"]]

# dummy encoding odor dilution
df2 = pd.get_dummies(df2, columns=['Odor_dilution'])

# extract molecular descriptors
class RDKit_2D:
    def __init__(self, smiles: pd.Series):
        self.smiles: pd.Series = smiles
        self.molecules: List[Chem.rdchem.Mol] = [Chem.MolFromSmiles(smile)
                                                 for smile in self.smiles]    

    def compute_descriptors(self) -> pd.DataFrame:
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
          [x[0] for x in Descriptors._descList]
        )
        names = calculator.GetDescriptorNames()
        values = [calculator.CalcDescriptors(molecule)              
                for molecule in self.molecules]
        df = pd.DataFrame(values, columns=names)
        df.insert(loc=0, column="SMILES", value=self.smiles.values)
        return df

data = pyrfume.load_data("keller_2016/molecules.csv", remote=True)
smiles = data.IsomericSMILES
kit = RDKit_2D(smiles)
df = kit.compute_descriptors()

# merging the df's using "SMILES"
X = pd.merge(df2, df, how = "left", on = "SMILES")

# saving y as pd.Series
file_y = open(r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\y.xlsx',
            'w+', newline='')
output_path_y = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\y.xlsx'

pd.Series(y).to_excel(output_path_y)

# saving X as df
file_X = open(r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\X.xlsx',
            'w+', newline='')
output_path_X = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\X.xlsx'

X.to_excel(output_path_X)
