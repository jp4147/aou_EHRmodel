import asyncio
import aiohttp
import json
import csv
from tqdm.asyncio import tqdm_asyncio
import openai
import pandas as pd

cancer_sites = [
 'skin', 
 'head and neck',
 'nasopharyngeal', 
 'salivary gland',
 'lymphoma/lymphoid',
 'leukemia',
 'multiple myeloma',
 'myelodysplastic syndromes',
 'anal',
 'breast',
 'lung/respiratory tract/pleura',
 'mesothelioma',
 'colon/rectal',
 'appendiceal',
 'cervical',
 'ovarian',
 'vulvar',
 'vaginal',
 'uterine/endometrial',
 'peritoneal',
 'esophagus',
 'stomach',
 'small intestine',
 'pancreas',
 'hepatobiliary',
 'gallbladder',
 'adrenal',
 'prostate',
 'male genital',
 'penile',
 'urinary tract/bladder',
 'brain',
 'eye/ocular',
 'neuroendocrine',
 'bone',
 'chordoma',
 'soft tissue sarcoma',
 'gastrointestinal stromal tumor',
 'germ cell tumor',
 'testicular',
 'choriocarcinoma',
 'placental',
 'wilms tumor',
 'hepatoblastoma',
 'kidney',
 'liver',
 'thyroid',
 'parathyroid',
 'pituitary',
 'histiocytosis',
 'myeloid/lymphoid neoplasms',
 'mast cell malignancy'
]

openai.api_key = "include your openai api key here"

# cat = pd.read_csv('additional_cancer_names.csv')
# cancer_sites = list(cat['category'].unique())
# cancer_sites.remove('other')

# malig = pd.read_csv('malignant443392.csv')
# cancer_names = list(set(list(malig['concept_name'])))

cancer_classification_results = {}

# OpenAI Categorization Function (Asynchronous) with Error Handling & Retries
async def categorize_cancer(session, name):
    prompt = f"""
        Categorize the following cancer name into one of these predefined categories: {", ".join(cancer_sites)}.

        **General Guidelines:**
        - Prioritize **tissue of origin** over anatomical location.
        - If tissue of origin is unclear, prioritize anatomical location.
        - For **small cell or large cell cancers**, categorize based on anatomical location.

        **Special Cases & Corrections:**
        - **Breast Tumors (including sarcomas within the breast):**  
        - If the cancer name includes "breast," categorize as **breast**, even if it is a sarcoma or undifferentiated tumor.
        - **Melanomas (including mucosal melanoma):**  
        - **Always classify melanoma as skin**, even if it is found in the oral cavity, mucosa, or other locations (e.g., tonsillar pillar, uvula, epiglottis, retromolar area).  
        - **Soft Tissue Sarcomas:**  
        - If the cancer is a **sarcoma (e.g., dendritic cell sarcoma, undifferentiated sarcoma, spindle cell sarcoma)**, classify as **soft tissue sarcoma**, except:
            - If it occurs in the **brain or brainstem**, classify as **brain**.
            - If it occurs in the **breast**, classify as **breast**.
        - **Kaposi’s sarcoma should always be classified as soft tissue sarcoma, regardless of primary anatomic site.**
        - **Histiocytosis & Histiocytic Disorders:**  
        - If the cancer is **histiocytosis** or contains **histiocytic elements**, categorize as **histiocytosis**.
        - **Mast Cell Malignancies:**  
        - If the cancer is a **mast cell tumor or mastocytosis**, categorize as **mast cell malignancy**.
        - **Pancreatic Tumors:**  
        - If a cancer is **specifically located in the pancreas or pancreatic duct**, categorize as **pancreas**, even if the tumor type is uncommon (e.g., Fibroblastic reticular cell tumor of islets of Langerhans, dendritic cell tumor of islets of Langerhans).
        - **Bone Tumors:**  
        - If the name includes **"bone," "rib," "sternum," "clavicle," "marrow"**, or a known bone tumor type (e.g., "Ewing sarcoma"), categorize as **bone**, unless explicitly soft tissue sarcoma.
        - **Head and Neck Cancers:**  
        - If **"nasopharyngeal," "tonsillar," "maxillary sinus," "subglottis"** appears, categorize as **head and neck**, **except melanoma**.
        - If the name includes **"larynx"**, categorize as **head and neck**.
        - **Skin Cancers:**  
        - Basal cell carcinoma (BCC) and squamous cell carcinoma (SCC) should always be classified as **skin**, regardless of location.
        - **Melanomas, including mucosal melanoma, should always be classified as skin** (e.g., melanoma of gum, epiglottis, retromolar area, uvula, tonsillar pillar).
        - **Do not classify "basal cell adenocarcinoma" as skin**—instead, return **remove.**
        - **Germ Cell Tumors:**  
        - If the name includes **"seminoma" or "germ cell"**, categorize as **germ cell tumor**, even if the location suggests another category.
        - **Leukemia & Myelodysplastic Syndromes:**  
        - If **"myelodysplastic," "refractory anemia," "excess blasts"** is present, categorize as **myelodysplastic syndromes**.  
        - If **"leukemia"** appears, classify as **leukemia**.
        - **Myeloproliferative Neoplasms (MPNs):**  
        - If **"myeloproliferative neoplasm"** appears, categorize as **myelodysplastic syndromes**, not myeloid/lymphoid neoplasms.
        - **Mesothelioma:**  
        - If the term **"mesothelioma"** is present, categorize as **mesothelioma**, even if it involves the pleura or lung.
        - **Gastrointestinal (GI) Tumors:**  
        - If the name includes **"ampulla of Vater," "gastric," "colorectal," "esophageal," or "pancreas"**, categorize under the corresponding GI organ.
        - **Cancers involving the eyelid, conjunctiva, or orbit:**  
        - Determine if they originate from **skin (BCC, SCC, melanoma)** or **eye structures (ocular melanoma, retinoblastoma).**  
        - Do **not** misclassify skin cancers near the eye as eye/ocular.
        
        **Additional Classification Rules:**
        - **Unclear Malignancy:** If it is **not explicitly malignant (cancerous)** or is a **benign hematologic condition**, categorize as **not cancer**.
        - **Secondary (Metastatic) Cancers:** If the term **"metastatic," "secondary," or "carcinomatosis"** appears, classify as **secondary**.
        
        **Other Rules:**
        - If the cancer does not fit any category, return **unknown**.
        - Provide **only** the category name in **lowercase** (e.g., "breast").

        cancer name: "{name}"  
        gpt category:
    """
    for attempt in range(3):  # Retry failed requests up to 3 times
        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai.api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are an expert in medical oncology classifications."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 20,
                    "temperature": 0
                }
            ) as response:
                res_json = await response.json()

                # Check if API response contains 'choices'
                if "choices" in res_json:
                    category = res_json["choices"][0]["message"]["content"].strip()
                    return name, category
                else:
                    error_msg = res_json.get("error", {}).get("message", "Unknown API Error")
                    print(f" API Error for '{name}': {error_msg}")
                    return name, "API_Error"

        except Exception as e:
            print(f" Exception for '{name}' on attempt {attempt + 1}: {str(e)}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff for retries

    return name, "Exception_Error"  # If all retries fail

async def batch_categorization(cancer_names, batch_size=50):
    global cancer_classification_results
    async with aiohttp.ClientSession() as session:
        tasks = []
        for name in cancer_names:
            tasks.append(categorize_cancer(session, name))

            # Process in batches
            if len(tasks) >= batch_size:
                results = await tqdm_asyncio.gather(*tasks)
                cancer_classification_results.update(dict(results))
                tasks = []

        # Process any remaining tasks
        if tasks:
            results = await tqdm_asyncio.gather(*tasks)
            cancer_classification_results.update(dict(results))

def save_results_json(filename="cancer_classification.json"):
    with open(filename, "w") as f:
        json.dump(cancer_classification_results, f, indent=4)
    print(f"Results saved to {filename}")

# Function to save results as CSV
def save_results_csv(filename="cancer_classification.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cancer name", "gpt category"])
        for name, category in cancer_classification_results.items():
            writer.writerow([name, category])
    print(f"Results saved to {filename}")

def cancer_category():
    return cancer_sites