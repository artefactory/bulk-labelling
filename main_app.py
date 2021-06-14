import streamlit as st

import lib.pages.page as bulk_labelling_app

def write_page(page):
    page.write()


def main():
    page = bulk_labelling_app
    write_page(page)

if __name__ == "__main__":
    st.set_page_config(layout='wide')
    main()
