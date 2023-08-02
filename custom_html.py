import streamlit as st
import streamlit.components.v1 as com

def custom_response(output):

    return f"""


        <div class="css-l3yxb1 ecja3eu1" style="background-color:rgb(240, 242, 246); border-radius: 8px; display:flex; margin-bottom:3%">
            <div class="answer-prompt" style="width:100%;margin-right: -5%;">
                <p style="padding: 0.5rem;margin: 0px;">{output}</p>
            </div>
            <div class="css-chk1w8 ecja3eu2" style="border-width: thick;border-color: black;margin: 3px;">
                <button title="Copy to clipboard" data-clipboard-text="{output}" class="css-10nk6p7 ecja3eu0" style="top: 0px; right: 0px;border: 0px;background-color: white;    border-radius: 8px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                </button>
            </div>
        </div>

    """


    
