import streamlit as st
import streamlit.components.v1 as components  
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import sklearn
estimators = sklearn.utils.all_estimators(type_filter=None)
import pandas as pd
import numpy as np
from pandasql import sqldf
import sweetviz as sv
import base64 
import tpot
from tabula import read_pdf
import html5lib
import requests
import time
import codecs
import os
import io
from sklearn.utils._testing import ignore_warnings
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder, JsCode
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html

timestr = time.strftime("%Y%m%d-%H%M%S")
st.set_page_config(page_title="Suite Analisi Dati", page_icon="🔍", layout='wide', initial_sidebar_state='auto')

st.markdown("<center><h1> Italian Intelligence Analytic Suite <small><br> Powered by INTELLIGENZAARTIFICIALEITALIA.NET </small></h1>", unsafe_allow_html=True)
st.write('<p style="text-align: center;font-size:15px;" > <bold>Tutti i tool di Analisi, Pulizia e Visualizzazione Dati in unico Posto <bold>  </bold><p><br>', unsafe_allow_html=True)



hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True, persist=True)
def reading_dataset():
    global dataset
    try:
        dataset = pd.read_excel(uploaded_file)
    except :
        dataset = pd.read_csv(uploaded_file)
        
    return dataset
		
def text_downloader(raw_text):
	b64 = base64.b64encode(raw_text.encode()).decode()
	new_filename = "Note_Analisi_IAITALIA_{}_.txt".format(timestr)
	st.sidebar.markdown("#### Download File ###")
	href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Scarica le tue NOTE !!</a>'
	st.sidebar.markdown(href,unsafe_allow_html=True)
	st.sidebar.subheader("")
	
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href
    
def app(dataset):
    # Use the analysis function from sweetviz module to create a 'DataframeReport' object.
    analysis = sv.analyze([dataset,'AnalisiEDA2_powered_by_IAI'], feat_cfg=sv.FeatureConfig(force_text=[]), target_feat=None)
    analysis.show_html(filepath='AnalisiEDA2_powered_by_IAI.html', open_browser=False, layout='vertical', scale=1.0)
    HtmlFile = open("AnalisiEDA2_powered_by_IAI.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code,height=1200, scrolling=True)
    st.markdown(get_binary_file_downloader_html('AnalisiEDA2_powered_by_IAI.html', 'Report'), unsafe_allow_html=True)
    st.success("Secondo Report Generato Con Successo, per scaricarlo clicca il Link quì sopra.")

def displayPDF(file):
    # Opening file from file path
    with open(file, "r") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display =F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'


############################################ANALYTIC SUITE
def AnalyticSuite()  :

	uploaded_file = st.file_uploader("Perfavore inserisci quì il file di tipo csv, usando come separatore la virgola!", type=["csv"])

	#st.sidebar.subheader("") 
	#st.sidebar.subheader("") 
	#st.sidebar.subheader("Notepad")
	#my_text = st.sidebar.text_area(label="Inserisci quì le tue osservazioni o note!", value="Al momento non hai nesuna nota...", height=30)

	#if st.sidebar.button("Salva"):
	#	text_downloader(my_text)
		
	if uploaded_file is not None:
	    dataset = pd.read_csv(uploaded_file)
	    colonne = list(dataset.columns)
	    options = st.multiselect("Seleziona le colonne che vuoi usare..",colonne,colonne)
	    dataset = dataset[options]
	    gb = GridOptionsBuilder.from_dataframe(dataset)

	    #customize gridOptions
	    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
	    gb.configure_grid_options(domLayout='normal')
	    gridOptions = gb.build()
	    
	    try:
	    	with st.expander("VISUALIZZA e MODIFICA il DATASET"):
	    		grid_response = AgGrid(
			    dataset, 
			    gridOptions=gridOptions,
			    update_mode="MODEL_CHANGED",
			    )
		
	    	with st.expander("VISUALIZZA delle STATISICHE di BASE"):
	    		st.write(dataset.describe())
	    except:
		    print("")
		 
	    st.markdown("", unsafe_allow_html=True)
	    task = st.selectbox("Cosa ti serve ?", ["Crea Report Personalizzato", "Scopri il Miglior Algoritmo di ML per i tuoi dati",
		                         "Crea PipeLine ADHOC in Python per i tuoi dati", "Utilizza le Query SQL sui tuoi dati","Pulisci i Miei Dati"])
		                         
		                         
	    if task == "Crea Report Personalizzato":
	    	if(st.button("Generami 2 Report Gratuitamente")):
	    		try :
			    	pr = ProfileReport(dataset, explorative=True, orange_mode=False)
			    	st_profile_report(pr)
			    	pr.to_file("AnalisiEDA_powered_by_IAI.html")
			    	st.markdown(get_binary_file_downloader_html('AnalisiEDA_powered_by_IAI.html', 'Report'), unsafe_allow_html=True)
			    	st.success("Primo Report Generato Con Successo, per scaricarlo clicca il Link quì sopra.")
			    	app(dataset)
			    	
			    	st.balloons()
	    		except Exception as e:
		    		print(e)
		    		st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova')

	    	
	    elif task == "Scopri il Miglior Algoritmo di ML per i tuoi dati":
	    	datasetMalgo = dataset
	    	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	    	datasetMalgo = datasetMalgo.select_dtypes(include=numerics)
	    	datasetMalgo = datasetMalgo.dropna()
	    	colonne = datasetMalgo.columns
	    	target = st.selectbox('Scegli la variabile Target', colonne )
	    	st.write("target impostato su " + str(target))
	    	datasetMalgo = datasetMalgo.drop(target,axis=1)
	    	colonne = datasetMalgo.columns
	    	descrittori =  st.multiselect('Scegli la variabili Indipendenti', colonne )
	    	st.write("Variabili Indipendenti impostate su  " + str(descrittori))
	    	
	    	problemi = ["CLASSIFICAZIONE", "REGRESSIONE" ]
	    	tipo_di_problema = st.selectbox('Che tipo di Algortimo devi utilizzare sui tuoi dati ?', problemi)
	    	percentuale_dati_test = st.slider('Seleziona la percentuale di dati per il Test', 0.1, 0.9, 0.25)
		
	    	X = dataset[descrittori]
	    	y = dataset[target]
		
	    	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=percentuale_dati_test)
		
	    	if(st.button("Svelami il Miglior Algoritmo per i miei dati Gratuitamente")):
	    		#try :
	    			
	    		if(tipo_di_problema == "CLASSIFICAZIONE"):

	    			with st.spinner("Dacci qualche minuto, stiamo provando tutti gli algoritmi di Classificazione sui tuoi dati"):
	    				clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
	    				models,predictions = clf.fit(X_train, X_test, y_train, y_test)
	    				st.write(models)
	    				models = pd.DataFrame(models)
	    				models.to_csv("MigliorAlgoritmo_powered_by_IAI.csv")
	    				st.markdown(get_binary_file_downloader_html('MigliorAlgoritmo_powered_by_IAI.csv', 'Rapporto Modelli Predittivi'), unsafe_allow_html=True)
	    				st.balloons()

	    		if(tipo_di_problema == "REGRESSIONE"):

	    			with st.spinner("Dacci un attimo, stiamo provando tutti gli algoritmi di Regressione sui tuoi dati"):
	    				reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
	    				models, predictions = reg.fit(X_train, X_test, y_train, y_test)
	    				st.write(models)
	    				models = pd.DataFrame(models)
	    				models.to_csv("MigliorAlgoritmo_powered_by_IAI.csv")
	    				st.markdown(get_binary_file_downloader_html('MigliorAlgoritmo_powered_by_IAI.csv', 'Rapporto Modelli Predittivi'), unsafe_allow_html=True)		
	    				st.balloons()
			    			
	    		#except Exception as e:
		    		#print(e)
		    		#st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova')
	    			
	    elif task == "Utilizza le Query SQL sui tuoi dati":
	    	q = st.text_input("Scrivi qui dentro la tua Query", value="SELECT * FROM dataset")
	    	if st.button("Applicami questa Query SQL sui miei dati Gratuitamente"):
	    		try :
		    		df = sqldf(q)
		    		df = pd.DataFrame(df)
		    		st.write(df)
		    		df.to_csv("RisultatiQuery_powered_by_IAI.csv")
		    		st.markdown(get_binary_file_downloader_html('RisultatiQuery_powered_by_IAI.csv', 'Riusltato Query Sql sui tuoi dati'), unsafe_allow_html=True)
		    		st.balloons()
		    	except Exception as e:
		    		print(e)
		    		st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova')
				
				
	    elif task == "Crea PipeLine ADHOC in Python per i tuoi dati":
	    	datasetPalgo = dataset
	    	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	    	datasetPalgo = datasetPalgo.select_dtypes(include=numerics)
	    	datasetPalgo = datasetPalgo.dropna()
	    	colonne = datasetPalgo.columns
	    	target = st.selectbox('Scegli la variabile Target', colonne )
	    	st.write("target impostato su " + str(target))
	    	datasetPalgo = datasetPalgo.drop(target,axis=1)
	    	colonne = datasetPalgo.columns
	    	descrittori =  st.multiselect('Scegli la variabili Indipendenti', colonne )
	    	st.write("Variabili Indipendenti impostate su  " + str(descrittori))
	    	
	    	problemi = ["CLASSIFICAZIONE", "REGRESSIONE" ]
	    	tipo = st.selectbox('Che tipo di Algortimo devi utilizzare sui tuoi dati ?', problemi)
	    	percentuale_dati_test = st.slider('Seleziona la percentuale di dati per il Test', 0.1, 0.9, 0.25)
		
	    	gen = st.slider('GENERAZIONI : Numero di iterazioni del processo di ottimizzazione della pipeline di esecuzione. Deve essere un numero positivo o Nessuno.', 1, 10, 5)
	    	pop = st.slider('POPOLAZIONE : Numero di dati da mantenere nella popolazione di programmazione genetica in ogni generazione.', 1, 150, 20)
	    	
	    	scor = ['accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'precision']
	    	sel_scor = st.selectbox('Che tipo di metrica vuoi che sia utilizzato ? Se non conosci questi metodi inserisci "accuracy"', scor)
		
	    	X = dataset[descrittori]
	    	y = dataset[target]
	    	
	    	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=percentuale_dati_test)
		
	    	if(st.button("Creami la miglior pipeline in Python Gratuitamente")):
	    		#try :
		    		
	    		if tipo=="CLASSIFICAZIONE":

	    			with st.spinner("Dacci qualche minuto, stiamo scrivendo il Codice in Python che implementa il miglior algoritmo sui tuoi dati e ottimizzandolo con gli iperparametri. Maggiore è il numero di Generazioni e Popolazione maggiore sarà il tempo di ATTESA..."):
	    				pipeline_optimizer = TPOTClassifier()
	    				pipeline_optimizer = TPOTClassifier(generations=gen, population_size=pop, scoring=sel_scor, cv=5,
							    random_state=42, verbosity=2)
	    				pipeline_optimizer.fit(X_train, y_train)
	    				#st.write(f"Accuratezza PIPELINE : {pipeline_optimizer.score(X_test, y_test)*100} %")
	    				pipeline_optimizer.export('PipelinePython_powered_by_IAI.py')
	    				filepipeline = open("PipelinePython_powered_by_IAI.py", 'r', encoding='utf-8')
	    				source_code = filepipeline.read() 
	    				st.subheader("Miglior PipeLine Rilevata Sui tuoi Dati ")
	    				my_text = st.text_area(label="Hai visto, Scriviamo anche il codice al posto tuo...", value=source_code, height=500)
	    				st.markdown(get_binary_file_downloader_html('PipelinePython_powered_by_IAI.py', 'Scarica il file python pronto per essere eseguito'), unsafe_allow_html=True)
	    				st.balloons()

	    		if tipo=="REGRESSIONE":

	    			with st.spinner(" Dacci qualche minuto, stiamo scrivendo il Codice in Python che implementa il miglior algoritmo sui tuoi dati e ottimizzandolo con gli iperparametri. Maggiore è il numero di Generazioni e Popolazione maggiore sarà il tempo di ATTESA..."):
	    				pipeline_optimizer = TPOTRegressor()
	    				pipeline_optimizer = TPOTRegressor(generations=gen, population_size=pop, scoring=sel_scor, cv=5,
							    random_state=42, verbosity=2)
	    				pipeline_optimizer.fit(X_train, y_train)
	    				#st.write(f"Accuratezza PIPELINE : {pipeline_optimizer.score(X_test, y_test)*100} %")
	    				pipeline_optimizer.export('PipelinePython_powered_by_IAI.py')
	    				filepipeline = open("PipelinePython_powered_by_IAI.py", 'r', encoding='utf-8')
	    				source_code = filepipeline.read() 
	    				st.subheader("Miglior PipeLine Rilevata Sui tuoi Dati ")
	    				my_text = st.text_area(label="Hai visto, Scriviamo anche il codice al posto tuo...", value=source_code, height=500)
	    				st.markdown(get_binary_file_downloader_html('PipelinePython_powered_by_IAI.py', 'Scarica il file python pronto per essere eseguito'), unsafe_allow_html=True)
	    				st.balloons()
			    
	    		#except Exception as e:
		    		#print(e)
		    		#st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova')
		    	
		    	
	    elif task == "Pulisci i Miei Dati":
	    	from datacleaner import autoclean
	    	dataset_pulito=dataset
	    	st.subheader("Ecco qualche INFO sul tuo Dataset Prima che venga pulito")
	    	import io 
	    	buffer = io.StringIO() 
	    	dataset.info(buf=buffer)
	    	s = buffer.getvalue() 
	    	with open("InformaziniDataset_powered_by_IAI.txt", "w", encoding="utf-8") as f:
	    	     f.write(s) 
	    	fileinfo = open("InformaziniDataset_powered_by_IAI.txt", 'r', encoding='utf-8')
	    	source_code = fileinfo.read() 
	    	st.text_area(label="info...", value=source_code, height=300)
	    	
	    	col1, col2, col3 = st.columns(3)
	    	
	    	if( col1.button("Pulisci i miei dati da Valori nulli o corrotti Gratuitamente")):
	    		try :
	    			with st.spinner(" Dacci qualche secondo per ripulire i tuoi dati da Valori NULL o NAN e quelli corrotti "):
			    	    	st.subheader("Ecco qualche INFO sul tuo Dataset Dopo essere stato Pulito")
			    	    	dataset_pulito=autoclean(dataset)
			    	    	buffer = io.StringIO() 
			    	    	dataset.info(buf=buffer)
			    	    	s = buffer.getvalue() 
			    	    	with open("InformaziniDataset_powered_by_IAI.txt", "w", encoding="utf-8") as f:
			    	    	     f.write(s) 
			    	    	fileinfo = open("InformaziniDataset_powered_by_IAI.txt", 'r', encoding='utf-8')
			    	    	source_code = fileinfo.read() 
			    	    	st.text_area(label="info dati puliti...", value=source_code, height=300)
			    	    	dataset_pulito.to_csv('DatasetPulito_powered_by_IAI.csv', sep=',', index=False)
			    	    	st.markdown(get_binary_file_downloader_html('DatasetPulito_powered_by_IAI.csv', 'Scarica i tuoi Dati puliti'), unsafe_allow_html=True)
			    	    	st.balloons()
	    		except Exception as e:
		    		print(e)
		    		st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova')
		    	    	
	    	if( col2.button("Normalizzami i valori Numerici [MINMAXSCALER] Gratuitamente")):
	    		try :
	    			with st.spinner(" Dacci qualche secondo per Normalizzare i tuoi dati "):
			    	    	datasetMM = dataset
			    	    	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
			    	    	datasetMM = datasetMM.select_dtypes(include=numerics)
			    	    	datasetMM = datasetMM.dropna()
			    	    	from sklearn.preprocessing import MinMaxScaler
			    	    	scaler = MinMaxScaler()
			    	    	scaled = scaler.fit_transform(datasetMM)
			    	    	colonneMM = datasetMM.columns
			    	    	scaled = pd.DataFrame(scaled, columns = colonneMM)
			    	    	st.write(scaled)
			    	    	scaled.to_csv('DatasetMINMAXSCALER_powered_by_IAI.csv', sep=',', index=False)
			    	    	st.markdown(get_binary_file_downloader_html('DatasetMINMAXSCALER_powered_by_IAI.csv', 'Dati normalizzati con metodo MINMAXSCALER by IAITALIA'), unsafe_allow_html=True)
			    	    	
			    	    	st.balloons()
	    		except Exception as e:
		    		print(e)
		    		st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova')
		    	    	
		    	    	
	    	if( col3.button("Standadizza i valori Numerici [STANDARSCALER] Gratuitamente ")):
	    		try :
	    			with st.spinner(" Dacci qualche secondo per Standardizzare i tuoi dati "):
			    	    	datasetSS = dataset
			    	    	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
			    	    	datasetSS = datasetSS.select_dtypes(include=numerics)
			    	    	datasetSS = datasetSS.dropna()
			    	    	from sklearn.preprocessing import StandardScaler
			    	    	scaler = StandardScaler()
			    	    	scaled = scaler.fit_transform(datasetSS)
			    	    	colonneSS = datasetSS.columns
			    	    	scaled = pd.DataFrame(scaled, columns = colonneSS)
			    	    	st.write(scaled)
			    	    	scaled.to_csv('DatasetSTANDARDSCALER_powered_by_IAI.csv', sep=',', index=False)
			    	    	st.markdown(get_binary_file_downloader_html('DatasetSTANDARDSCALER_powered_by_IAI.csv', 'Dati normalizzati con metodo STANDARSCALER by IAITALIA'), unsafe_allow_html=True)
			    	    	
			    	    	st.balloons()
	    		except Exception as e:
		    		print(e)
		    		st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova')



###########################WEBSCRAPESUITE
def ScrapeSuite():

	#st.sidebar.subheader("") 
	#st.sidebar.subheader("") 
	#st.sidebar.subheader("Notepad")
	#my_text = st.sidebar.text_area(label="Inserisci quì le tue osservazioni o note!", value="Al momento non hai nesuna nota...", height=30)

	#if st.sidebar.button("Salva"):
	#	text_downloader(my_text)
		
	st.subheader("") 
	st.markdown("### **1️⃣ Inserisci l'url di una pagina web contenente almeno una Tabella **")
	
	try:
	    url =  st.text_input("", value='https://www.tuttosport.com/live/classifica-serie-a', max_chars=None, key=None, type='default')
	    if url and st.button("Cercami le tabelle nella pagina web per poi scaricarle Gratis"):
	    	try :
	    		
		    	arr = ['https://', 'http://']
		    	if any(c in url for c in arr):
		    	    header = {
			    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
			    "X-Requested-With": "XMLHttpRequest"
			    }

		    	    @st.cache(persist=True, show_spinner=False)
		    	    def load_data():
		    	        r = requests.get(url, headers=header)
		    	        return pd.read_html(r.text)

		    	    df = load_data()

		    	    length = len(df)
			    
		    	    if length == 1:
		    	        st.write("Questa Pagina Web Contiene una sola Pagina Web" )
		    	    else: st.write("Questa Pagina Web Contiene " + str((length-1)) + " Tabelle" )
		    	    st.balloons()

		    	    st.subheader("") 
		    	    def createList(r1, r2): 
		    	        return [item for item in range(r1, r2)] 
			       
		    	    r1, r2 = 0, length
		    	    funct = createList(r1, r2)
		    	    st.markdown('### **2️⃣ Seleziona la tabella che desideri esportare **')
		    	    for i in funct:
			    ###### Selectbox - Selectbox - Selectbox - Selectbox - Selectbox - Selectbox - Selectbox -
		    	    	with st.expander("Tabella numero : " + str(i)): 
		    	    		df1 = df[i]
		    	    		if df1.empty:
		    	    			st.warning ('ℹ️ - Mannaggia! Qualcosa è andato storto..')
		    	    		else:
		    	    			df1 = df1.replace(np.nan, 'empty cell', regex=True)
		    	    			st.dataframe(df1)
		    	    			try:
	    	    					nome_web=csv = "web_table_"+str(i)+".csv"
	    	    					csv = df1.to_csv(index=False)
	    	    					b64 = base64.b64encode(csv.encode()).decode()	
	    	    					st.markdown('### ** ⬇️ Scarica la tabella in formato csv **')
	    	    					href = f'<a href="data:file/csv;base64,{b64}" download="web_table.csv">** Clicca Qui per Scaricare la Tabella! 🎉**</a>'
	    	    					st.markdown(href, unsafe_allow_html=True)
	    	    				
		    	    			except ValueError:
		    	    				pass
	    	
		    	else:
		    		st.error ("⚠️ - L'URL deve avere un formato valido, Devi iniziare con *https://* o *http://*")    
    		except Exception as e:
		    	print(e)
		    	st.error('Mannaggia, ci dispiace qualcosa non è andato come doveva, riprova')
	except ValueError:
	    st.info ("ℹ️ - Non abbiamo trovato tabelle da Esportare ! 😊")



###########################PDFTOCSV
def pdftocsv():
	st.subheader("") 
	st.markdown("### **1️⃣ Carica il Tuo PDF **")
	uploaded_file = st.file_uploader('Scegli un file con estensione .pdf contenente almeno una tabella', type="pdf")
	if uploaded_file is not None:
		#displayPDF(uploaded_file)
		if st.button("Trovami tutte le tabelle Gratis "):
			try :

				#df = read_pdf(uploaded_file, pages='all')[0]
				tables = read_pdf(uploaded_file, pages='all')
				j=0
				for tabelle in tables :
					try:
						if not tabelle.empty :
							j=j+1
							print(tabelle)
							with st.expander("Tabella numero : " + str(j) ):
								df_temp = pd.DataFrame(tabelle)
								df_temp = df_temp.dropna()
								st.write(df_temp)
								csv = df_temp.to_csv(index=False)
								b64 = base64.b64encode(csv.encode()).decode()
								if st.button("Scarica Tabella numero " + str(j)):
									st.markdown('### ** ⬇️ Scarica la tabella in formato csv **')
									href = f'<a href="data:file/csv;base64,{b64}" download="PDF_table{str(j)}.csv">** Clicca Qui per Scaricare il Tuo Dataset! 🎉**</a>'
									st.markdown(href, unsafe_allow_html=True)
	
					except ValueError:
						pass
			except ValueError:
				st.info ("ℹ️ - Non abbiamo trovato tabelle da Esportare ! 😊")



def NuovoExcel():

	operazione = st.selectbox("Cosa ti serve ?", ["Un foglo vuoto", "Partire dal mio set di dati"])
	if( operazione == "Un foglo vuoto"):
		df = pd.DataFrame(
		    "",
		    index=range(100000),
		    columns=list("abcdefghilmnopq"),
		)

		gb = GridOptionsBuilder.from_dataframe(df)
		gb.configure_default_column(editable=True)

		gb.configure_grid_options(enableRangeSelection=True)
		with st.spinner('Aspetta un attimo...'):
			response = AgGrid(
			    df,
			    height=800, 
			    width='100%',
			    gridOptions=gb.build(),
			    fit_columns_on_grid_load=True,
			    allow_unsafe_jscode=True,
			    enable_enterprise_modules=True
			)
	if( operazione == "Partire dal mio set di dati"):
		uploaded_file_2 = st.file_uploader("Perfavore inserisci quì il file di tipo csv, usando come separatore la virgola!", type=["csv"])

		if uploaded_file_2 is not None:
			dataset = pd.read_csv(uploaded_file_2)
			colonne = list(dataset.columns)
			options = st.multiselect("Seleziona le colonne che vuoi usare..",colonne,colonne)
			dataset = dataset[options]
			gb = GridOptionsBuilder.from_dataframe(dataset)
			gb.configure_default_column(editable=True)

			gb.configure_grid_options(enableRangeSelection=True)
			with st.spinner('Aspetta un attimo...'):
				response = AgGrid(
				    dataset,
				    height=800, 
				    width='100%',
				    gridOptions=gb.build(),
				    fit_columns_on_grid_load=True,
				    allow_unsafe_jscode=True,
				    enable_enterprise_modules=True
				)



def convertiExcel():

	operazione1 = st.selectbox("Cosa ti serve ?", ["Da .xls a .csv", "Da .csv a .xls"])
	if( operazione1 == "Da .xls a .csv"):
		uploaded_file_2 = st.file_uploader("Perfavore inserisci quì il file di tipo .xls ", type=["xls"])

		if uploaded_file_2 is not None:
			df = pd.read_excel(uploaded_file_2)
			towrite = io.BytesIO()
			downloaded_file = df.to_csv (towrite, index = None, header=True)
			towrite.seek(0)  # reset pointer
			b64 = base64.b64encode(towrite.read()).decode()  # some strings
			linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="DatasetConvertito.csv">Scarica il Datset Convertito in .csv</a>'
			st.markdown(linko, unsafe_allow_html=True)

	if( operazione1 == "Da .csv a .xls"):
		uploaded_file_2 = st.file_uploader("Perfavore inserisci quì il file di tipo .csv ", type=["csv"])

		if uploaded_file_2 is not None:
			df = pd.read_csv(uploaded_file_2)
			towrite = io.BytesIO()
			downloaded_file = df.to_excel (towrite, encoding='utf-8', index = None, header=True)
			towrite.seek(0)  # reset pointer
			b64 = base64.b64encode(towrite.read()).decode()  # some strings
			linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="DatasetConvertito.xls">Scarica il Datset Convertito in .xlx</a>'
			st.markdown(linko, unsafe_allow_html=True)


#################MAIN

def main():

	
	Menu = option_menu("La miglior Suite DataScience 🐍🔥", ["Analytic Suite", "WebScrape Siute", "Da pdf a Csv", "Da excel a csv", "Excel Online"],
				 icons=['clipboard-data', 'globe', 'file-pdf', 'file-earmark-spreadsheet'],
				 menu_icon="app-indicator", default_index=0,orientation='horizontal',
				 styles={
"container": {"padding": "5!important", "background-color": "#fafafa", "width": "100%"},
"icon": {"color": "blue", "font-size": "15px"}, 
"nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
"nav-link-selected": {"background-color": "#02ab21"},
}
)

	if Menu == "Analytic Suite" :
		AnalyticSuite()
	if Menu == "WebScrape Siute" :
		ScrapeSuite()
	if Menu == "Da pdf a Csv" :
		pdftocsv()
	if Menu == "Excel Online" :
		NuovoExcel()
	if Menu == "Da excel a csv" :
		convertiExcel()


	
    			
main()
		
