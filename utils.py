import pandas as pd
import datetime
from io import BytesIO
import re

# Compute average of pulled CID confidence scores
def compute_average_score(docs):
    total = 0
    for doc in docs.values():
        total += doc.score
        avgscore = total / len(docs) ## convert total score to avg
    
    avgscore *= 100 ## convert from decimal to percentage
    return avgscore

# Formate SME names for display
def parse_sme_name(sme):
    if sme == "N/A":
        return "<insert name here>"
    
    # Split name, remove whitespace
    name_list = sme.replace('/', ',').split(',')
    name_list = [name.strip() for name in name_list]
    l = len(name_list)

    # Reorder
    if l > 2:  ## Handle multiple names case
        firstnames = [name_list[i] for i in range(1, len(name_list)-1)]
        firstname = ' '.join(firstnames)
        middlename = name_list[-1].replace('(', '\"').replace(')', '\"').strip()
        lastname = name_list[0]
        fullname = firstname + ' ' + middlename + ' ' + lastname
    elif l == 2:  ## Handle Firstname Lastname case
        firstname = name_list[1].replace('(', '\"').replace(')', '\"').strip()
        lastname = name_list[0]
        fullname = firstname + ' ' + lastname
    elif l == 1:
        sme = "STRANGE SME NAME. INSPECT ORIGINAL DOCUMENT"
    else:
        fullname = name_list[1] + ' ' + name_list[0]
    return fullname

# Read questions from file and gather row data
def read_questions(file):

    # Read CSV or Excel file into a Pandas DataFrame with no headers
    if file.type == 'text/csv':
        df = pd.read_csv(file, header=None)
    elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        df = pd.read_excel(file, header=None)
    else:
        return [], 2
    
    # Extract non-empty cells as questions and gather row data
    questions = []
    rows = []
    for _, row in df.iterrows():
        question = ""
        for cell in row:
            if pd.notna(cell):
                question += (str(cell).strip() + " ") ## Concatenate cells in a row
        if question != "": ## If cell not empty
            questions.append(question) ## Add questions in row
            rows.append(row) ## Add row data
    if questions==[]: ## If not question
        return [], 1, []
    return questions, 0, rows

# Format excel sheet considering original rows
def to_excel(df, rows):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    workbook = writer.book
    worksheet = workbook.add_worksheet("Sheet1")
    worksheet = writer.sheets['Sheet1']
    format = workbook.add_format({'bold': True}) 
    title = ["Question", "Answer", "Confidence", "SME", "CIDs"]
    n = 0
    k = 0
    for i in title:
        worksheet.write(0, k, i, format)
        k += 1
    for row in rows:
        for column in range(df.shape[1]):
            worksheet.write(row.name + 1, column, str(df.iloc[n,column]))
        n += 1
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# Compare dates
def getMostRecentDate(x, y):

    # Convert strings to datetime objects
    date_x = datetime.strptime(x, "%Y-%m-%d %H:%M:%S %z")
    date_y = datetime.strptime(y, "%Y-%m-%d %H:%M:%S %z")

    # Compare dates and print the more recent one
    if date_x > date_y:
        return date_x
    else:
        return date_y

# Convert dataframe to html table with hyperlinks
def to_html(df, cids):
    n = 0
    for cid in cids:
        links = df["Source Links"].iloc[n] ## Get relevant links
        k = 0
        for link in links:
            links[k] = f'<a target="_blank" href="{link}">{cid[k]}</a>' ## Convert to html hyper links
            k += 1
        df["Source Links"].iloc[n] = links ## Reinsert links
        n += 1
    return df

# Convert links to hyperlinks for excel sheet
def to_hyperlink(df, cids):
    cols = {}
    n = 0
    for cid in cids:
        links = df["Source Links"].iloc[n] ## Get relavent links for row
        k = 0
        for link in links:
            # Convert links into hyperlinks for excel format
            links[k] = f'=HYPERLINK("{link}", "{cid[k]}")'

            # Create a dictionary to track which links go into which columns
            if cols.get(k) != None:
                cols[k].append(links[k])
            else:
                cols[k] = [links[k]]
            k += 1

        # If column list is not long enough to fit append with '-'
        while k <= 4:
            if cols.get(k) != None:
                cols[k].append('-')
            else:
                cols[k] = ['-']
            k += 1
        n += 1

    # Insert column lists from dictionary into columns for excel
    for col in range(len(cols.keys())):
        df.insert(4 + col, f'Link {col}', cols[col], False)

    # Drop the source links column
    df.drop(df.columns[9], axis=1, inplace=True)
    return df

# Get relevant SMEs for unanswered questions
def get_SMEs(df):
    unanswered = {}
    for i, row in df.iterrows():

        # If answer contains key word or key phrase
        if re.search(r"sorry|Sorry", df["Answer"][i]) or re.search(r"(not|no)?.*?(not|no|contain|clear|provided|provide) (enough )?(information|context|answer)", df["Answer"][i]) or re.search(r"answer.*?(not clear|unclear)", df["Answer"][i]) or re.search(r"call failed", df["Answer"][i]) is not None:
            
            # Add relavant SME to dictionary with questions
            if unanswered.get(df["SMEs"][i]) != None:
                unanswered[df["SMEs"][i]].append(f'Question {i}: {df["Question"][i]}')
            else:
                unanswered[df["SMEs"][i]] = [f'Question {i}: {df["Question"][i]}']
    return unanswered

def get_email_text(query, best_sme):

    print("Drafting email...")
    # Get response from OpenAI
    # prompt = f"Please write a brief and professional business email to someone named {best_sme} asking {query}. Include only the text of the email in your response, not any sort of email address, and should be formatted nicely. The email should start with Subject: __ \n\nand end with the exact string \n\n'[Your Name]'."
    # response = openai.Completion.create(
    #     engine='immerse-3-5',
    #     prompt=prompt,
    #     temperature=0.3,
    #     max_tokens=400,
    #     frequency_penalty=0.0,
    #     presence_penalty=0,
    # )
    # Substitute sections of email text and write
    # email_response = response.choices[0].text
    # subject_index = email_response.find("Subject:")
    # name_index = email_response.find("[Your Name]")
    # email_response = email_response[subject_index:name_index+len("[Your Name]")].strip()
    # email_content.write(email_response)

    email_text = f"""Subject: [subject of question]

    Good [morning/afternoon] {best_sme},

    We are working on a Security request for one of our customers. Their Technical Questionnaire has a question(s) for which you are listed as the subject matter expert. Please provide a response to the following question, after which we will add it to our database of standard enterprise content.

    

    {query}

    

    Our time is limited. Please provide a response by no later than [time], [Day], [mm/dd].

    If you require more time, or if the question(s) need to be redirected please respond upon receipt of this email so we have adequate time to redirect to the appropriate person(s).

    If you have questions or require additional information about the customer, products and/or services please reach out.

    Kind Regards,

    

    [Your Name]"""
    return email_text