import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

print("Duke lexuar datasetin e pastruar 'heart_cleaned.csv'...")
df_cleaned = pd.read_csv("heart_cleaned.csv")

# Largojmë kolonat numerike kontinue (mosha, bmi, etj.) që shoqja i ka standardizuar
numeric_continuous = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
                      'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level']
df_apriori = df_cleaned.drop(columns=[col for col in numeric_continuous if col in df_cleaned.columns])

# Kthejmë të gjitha kolonat kategorike të mbetura në Boolean (True/False)
for col in df_apriori.columns:
    df_apriori[col] = df_apriori[col].astype(str).str.strip().str.lower().isin(['true', '1', '1.0', 'yes'])

print("\nDuke ekzekutuar Algoritmin Apriori (Min Support = 10%)...")
frequent_itemsets = apriori(df_apriori, min_support=0.1, use_colnames=True)

print("Duke gjeneruar Rregullat e Asociacionit (Min Confidence = 50%)...")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Renditja e rregullave sipas fuqisë (lift)
rules = rules.sort_values(by='lift', ascending=False)

print(f"\nU gjetën gjithsej {len(rules)} rregulla të asociacionit!")
print("\nTOP 10 RREGULLAT MË TË FORTA")

# Formatimi që të printohen bukur në ekran pa kthesa e kllapa {}
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_string())

# Ruajtja në CSV për Excel
rules.to_csv("association_rules_results.csv", index=False)
print("\nTabela u ruajt në skedarin 'association_rules_results.csv'!")