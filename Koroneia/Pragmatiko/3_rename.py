import sys
import io

# Αλλαγή της προεπιλεγμένης κωδικοποίησης εξόδου σε UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd

# Ορισμός της διαδρομής του φακέλου και του αρχείου Excel στον τρέχοντα φάκελο
current_dir = os.getcwd()
folder_path = os.path.join(current_dir, "jpgs")
excel_path = os.path.join(folder_path, "ocr.xlsx")

# Ανάγνωση του Excel αρχείου
df = pd.read_excel(excel_path)

# Εκτύπωση των ονομάτων των στηλών για να δούμε πώς αναγνωρίζονται
print("Ονόματα στηλών στο Excel:", df.columns.tolist())

# Αφαίρεση περιττών κενών από τα ονόματα των στηλών
df.columns = df.columns.str.strip()

# Επανεκτύπωση των ονομάτων στηλών μετά την αφαίρεση κενών για επιβεβαίωση
print("Ονόματα στηλών μετά την αφαίρεση κενών:", df.columns.tolist())

# Έλεγχος τύπων δεδομένων για να δούμε αν υπάρχουν δεκαδικοί αριθμοί
print("Τύποι δεδομένων:", df.dtypes)

# Βρόχος για τη μετονομασία των αρχείων
for index, row in df.iterrows():
    try:
        # Κατασκευή του παλιού ονόματος αρχείου (π.χ., frame_0.jpg)
        old_name = f"frame_{int(row['File Number'])}.jpg"
        
        # Κατασκευή του νέου ονόματος αρχείου με σωστή μετατροπή σε ακέραιους
        new_name = f"{int(row['Year'])}_{int(row['Month']):02d}_{int(row['Day']):02d}.jpg"
        
        # Πλήρεις διαδρομές για τη μετονομασία
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        
        # Έλεγχος αν το παλιό αρχείο υπάρχει πριν τη μετονομασία
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Μετονομάστηκε: {old_name} -> {new_name}")
        else:
            print(f"Το αρχείο δεν βρέθηκε: {old_name}")
    except KeyError as e:
        print(f"Σφάλμα με τη στήλη: {e}")
    except Exception as e:
        print(f"Άλλο σφάλμα: {e}")

print("Η διαδικασία μετονομασίας ολοκληρώθηκε.")
