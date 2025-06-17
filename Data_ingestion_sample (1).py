import os
import re
import nltk
from docx import Document
from pypdf import PdfReader 
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain.vectorstores import AzureSearch
from langchain.document_loaders import PyPDFLoader , DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader



def read_pdf(path):
    reader = PdfReader(path) 
    pdf_content=""

    for i in range(len(reader.pages)):
        page = reader.pages[i]
        pdf_content+=page.extract_text()+"\n"
    return pdf_content


def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def read_file(folder_path):
    print(os.listdir(folder_path))
    if len(os.listdir(folder_path))==0:
        raise ValueError("No File Found")
    else:
        flag_relevant_file=0
        content_list=[]
        for file_name in os.listdir(folder_path):
            file_extension = os.path.splitext(file_name)[1]
            file_path=folder_path+'/'+file_name
            if file_extension == '.pdf':
                content = read_pdf(file_path)
                content_list.append(content)
                flag_relevant_file=1
            elif file_extension == '.docx':
                content=read_docx(file_path)
                content_list.append(content)
                flag_relevant_file=1
        if flag_relevant_file==0:
            raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")
        else:
            return content_list


def remove_newlines(serie):
   serie = serie.replace('\n', '. ')
   serie = serie.replace('\\n', '. ')
   serie = serie.replace('  ', ' ')
   serie = serie.replace('  ', ' ')
   return serie


def count_tokens(text):
    return (len(nltk.word_tokenize(text)))

def split_text_into_chunks(text, max_tokens):
    sentences = text.split('.')
    chunks = []
    donot_have_dots=0
    for c in sentences:
        # print(count_tokens(c))
        if count_tokens(c)>1000:
            donot_have_dots=1
            print("Dot separator not applied")
            break
    if donot_have_dots==0:
        current_chunk = ""
        token_count = 0

        for sentence in sentences:
            # Add sentence to current chunk
            current_chunk += sentence + '.'

            # Count tokens in current chunk
            token_count += count_tokens(sentence)

            # If token count exceeds max_tokens, add current chunk to chunks list
            if token_count >= max_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                token_count = 0

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    else:
        sentences = text.split()
        current_chunk = ""
        token_count = 0

        for sentence in sentences:
            # Add sentence to current chunk
            current_chunk += sentence + ' '

            # Count tokens in current chunk
            token_count += count_tokens(sentence)

            # If token count exceeds max_tokens, add current chunk to chunks list
            if token_count >= max_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                token_count = 0

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks


def write_string_to_txt(input_string, file_path):
    with open(file_path, 'w') as file:
        file.write(input_string)

def get_llm():
    llm =  AzureChatOpenAI(
        openai_api_version="",
        deployment_name="",
        openai_api_key="",
        azure_endpoint="",
        temperature=0,n=5)
    
    return llm

def preprocessing_content(content):
    preprocessed_content=[]
    for c in content:
        preprocessed_content.append(remove_newlines(c))
    return preprocessed_content


def get_chunks(preprocessed_content,max_tokens):
    final_chunks=[]
    for i in range(len(preprocessed_content)):
        chunks = split_text_into_chunks(preprocessed_content[i], max_tokens)
        final_chunks.append(chunks)
    return final_chunks


def get_prod_nav_data(llm,final_chunks):
    product_info=""
    navigation_info=""
    print("Data Travelling towards prompt.")
    for first_index in range(len(final_chunks)):
        for second_index in range(len(final_chunks[first_index])):
            info=final_chunks[first_index][second_index]
            prompt_product = f"""Please retrieve detailed information about a product from the provided information.

            - Examine thoroughly for all the elements and features of the product.
            - Specify functionalities of each element that you are able to identify in the product.
            - The output should only contain your findings about the product in paragraph form.
            - The output should strictly contain most relevant, crisp and concise findings in maximum 2 lines in paragraph form.
            - Please do not include any personal identificable information in the response, use generic terms in place.

            Information : {info}
            """

            prompt_navigation = f"""You are an intelligent navigation steps finder who finds navigation steps from the given information. You have been given information for a specific product. If the information includes navigation, such as transitioning between UI pages or utilizing product features, return those navigation steps. If no navigation information is present in the information, return null.

            - Just provide navigation steps in a paragraph form in maximum three lines by utilising the information present in the information.
            - If you don't find any relevant steps or navigation information, just return null. Do not try to make up steps.
            - The response should strictly be in paragraph form only.

            Information : {info}
            """
            prod_ans = llm.predict(prompt_product)
            print("called")
            product_info +=  prod_ans + "\n"

            nav_ans=llm.predict(prompt_navigation)
            if nav_ans != "null":
                navigation_info += nav_ans + "\n"

    return product_info,navigation_info
def push_to_indexes(temp_prod_path,temp_nav_path,chunk_size_prod,chunk_size_nav,overlap_prod,overlap_nav,index_name_product,index_name_nav):

    os.environ["AZURE_OPENAI_API_KEY"] = ""
    os.environ["AZURE_OPENAI_ENDPOINT"] = ""


    embeddings = AzureOpenAIEmbeddings(
    azure_deployment="embedding",
    openai_api_version="2023-05-15",
    )


    # Set up the connection
    service_endpoint = ""
    api_key = ""

    acs_prod = AzureSearch(azure_search_endpoint = service_endpoint, azure_search_key = api_key, index_name = index_name_product, 
                    embedding_function = embeddings.embed_query)

    acs_nav = AzureSearch(azure_search_endpoint = service_endpoint, azure_search_key = api_key, index_name = index_name_nav, 
                    embedding_function = embeddings.embed_query)


    loader_nav = TextLoader(temp_nav_path,autodetect_encoding=True)
    documents_nav = loader_nav.load()
    text_splitter_nav = TokenTextSplitter(chunk_size=chunk_size_nav, chunk_overlap=overlap_nav)
    docs_nav =text_splitter_nav.split_documents(documents_nav)

    loader_prod = TextLoader(temp_prod_path,autodetect_encoding=True)
    documents_prod = loader_prod.load()
    text_splitter_prod = TokenTextSplitter(chunk_size=chunk_size_prod, chunk_overlap=overlap_prod)
    docs_prod =text_splitter_prod.split_documents(documents_prod)

    #pushing the data to the vectorbase

    acs_nav.add_documents(documents=docs_nav)
    acs_prod.add_documents(documents=docs_prod)

    print("----------------------------------------------")
    print("Data Successfully pushed to Indexes.")
################




def add_or_update_document(document, index_name):
    """

    Adds or updates a document in the specified Azure Search index.

    Args:
        document (dict): The document to be upserted. Must include the key field defined in the index.
        index_name (str): The name of the Azure Search index.

    Raises:
        ValueError: If the required environment variables for the endpoint or API key are not set.
    """
    # Retrieve the Azure Search endpoint and key from environment variables
    
    os.environ["AZURE_OPENAI_API_KEY"] = ""
    os.environ["AZURE_OPENAI_ENDPOINT"] = ""


    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="embedding",
        openai_api_version="2023-05-15",
    )
   
    service_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    api_key = os.environ.get("AZURE_SEARCH_KEY")
    if not service_endpoint or not api_key:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY must be set in environment variables.")
    
    # Create the Azure Search client using the endpoint and key
    search_client = AzureSearch(azure_search_endpoint = service_endpoint, azure_search_key = api_key, index_name = index_name, 
                    embedding_function = embeddings.embed_query)

    # Upsert (add or update) the document using add_documents.
    # This method updates the document if it exists, or adds it if it doesn't.
    result = search_client.add_documents(documents=[document])
    
    print("Document add/update operation result:", result)


################

def main():
    max_tokens = 2000
    Data_local_path='Files'
    temp_prod_path = "temp/product_data.txt"
    temp_nav_path="temp/navigation_data.txt"
    chunk_size_prod=1000
    chunk_size_nav=500
    overlap_prod=100
    overlap_nav=50
    index_name_product = "test_prod"
    index_name_nav = "test_nav"

    content=read_file(Data_local_path)
    preprocessed_content=preprocessing_content(content)
    final_chunks=get_chunks(preprocessed_content,max_tokens)

    print("[")
    for i in final_chunks:
        print("[")
        for j in i:
            print(count_tokens(j),",")
        print("]")
    print("]")

    llm=get_llm()
    product_info,navigation_info=get_prod_nav_data(llm,final_chunks)

    print("----------------------Product Info--------------------------------")
    print(product_info)
    print("----------------------Navigation Info---------------------------")
    print(navigation_info)

    write_string_to_txt(product_info, temp_prod_path)
    write_string_to_txt(navigation_info,temp_nav_path)

    push_to_indexes(temp_prod_path,temp_nav_path,chunk_size_prod,chunk_size_nav,overlap_prod,overlap_nav,index_name_product,index_name_nav)


if __name__ == '__main__':
    main()