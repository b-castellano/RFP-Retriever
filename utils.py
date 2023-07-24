import pandas as pd
import datetime

def compute_average(ids, scores):
    total = 0
    for id in ids:
        id = id.strip()
        total += scores[id]
    if len(ids) > 0:
        avgscore = total / len(ids)     # convert total score to avg
    else:
        avgscore = 0
    avgscore *= 100                 # convert from decimal to percentage
    return avgscore

def parse_sme_name(sme):
    if sme == "N/A":
        return "<insert name here>"
    # Split name, remove whitespace
    name_list = sme.replace('/', ',').split(',')
    name_list = [name.strip() for name in name_list]
    l = len(name_list)
    # Reorder
    if l > 2:  # handle multiple names case
        firstnames = [name_list[i] for i in range(1, len(name_list)-1)]
        firstname = ' '.join(firstnames)
        middlename = name_list[-1].replace('(', '\"').replace(')', '\"').strip()
        lastname = name_list[0]
        fullname = firstname + ' ' + middlename + ' ' + lastname
    elif l == 2:  # handle Firstname Lastname case
        firstname = name_list[1].replace('(', '\"').replace(')', '\"').strip()
        lastname = name_list[0]
        fullname = firstname + ' ' + lastname
    elif l == 1:
        sme = "STRANGE SME NAME. INSPECT ORIGINAL DOCUMENT"
    else:
        fullname = name_list[1] + ' ' + name_list[0]
    return fullname

def remove_duplicates(original, arr1, arr2=[], arr3=[]):
    index_dict = {}
    # Find indexes of duplicates in original array
    for i, x in enumerate(original):
        if x not in index_dict:
            index_dict[x] = [i]
        else:
            index_dict[x].append(i)
    # Remove corresponding items from other arrays
    for indexes in index_dict.values():
        if len(indexes) > 1:
            for i in indexes[1:]:
                if i < len(arr1):
                    del arr1[i]
                if i < len(arr2):
                    del arr2[i]
                if i < len(arr3):
                    del arr3[i]
    unique = list(set(original))
    original = [x for x in unique]
    return original, arr1, arr2, arr3

def read_questions(file):
    # Read CSV or Excel file into a Pandas DataFrame with no headers
    if file.type == 'text/csv':
        df = pd.read_csv(file, header=None)
    elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        df = pd.read_excel(file, header=None)
    else:
        return [], 2
    # Extract non-empty cells as questions and append question marks
    questions = []
    for _, row in df.iterrows():
        for cell in row:
            if pd.notna(cell):
                question = str(cell).strip()
                if not question.endswith('?'):
                    question += '?'
                questions.append(question)
    if questions==[]:
        return [], 1
    return questions, 0


def getMostRecentDate(x, y):
    # Convert strings to datetime objects
    date_x = datetime.strptime(x, "%Y-%m-%d %H:%M:%S %z")
    date_y = datetime.strptime(y, "%Y-%m-%d %H:%M:%S %z")

    # Compare dates and print the more recent one
    if date_x > date_y:
        return date_x
    else:
        return date_y