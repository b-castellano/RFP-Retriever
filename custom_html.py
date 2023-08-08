import streamlit as st
import streamlit.components.v1 as com
import math

### This code sets the blue box & copy button in the single-response case for the front-end

def helper(output):
    # Get number of pixels necessary to display 
    num_char = len(output)
    num_lines = math.ceil(num_char/97)
    h = num_lines * 20 + 30
    return com.html(
        f"""
        <div class="css-l3yxb1 ecja3eu1" style="background-color:rgba(218,246,250,255); border-radius: 8px; display:flex; margin-bottom:3% padding:15px;">
            <div class="answer-prompt" id="copy-input" style="width:100%;margin-right: -5%;z-index:10;display:flex;align-items:center;">
                <p style="padding:0.5rem;margin: 0px;flex-grow:1; font-family:'Streamlit';">{output}</p>
                <button title="Copy to clipboard" id='text' onclick='navigator.clipboard.writeText("{output}")' type='button' class="css-10nk6p7 ecja3eu0" style="border: 0px;background-color: white; border-radius: 8px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                </button>
            </div>
        </div>
        """,
        height=h
    )

def custom_response(output):
    return f"""
    <style>
        .hide {{ opacity: 0; }}
        .answer-prompt:hover + .hide {{ opacity: 1; transition: all 1s; }}
        p {{ font-family:'Streamlit'; }}
    </style>
    {helper(output)}
    """