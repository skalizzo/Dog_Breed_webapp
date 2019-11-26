import pandas as pd
import sqlite3

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ExtractData(metaclass=Singleton):
    def __init__(self):
        self.pfadlokal = 'C:/VODTagesdaten_lokal/DB'
        self.data = None
        self.dbpfad_amazon = "G:/Digitale Distribution/10_PLATTFORMEN/16_AMAZON/99_Abrechnung/Daily Reports/Datenbank/DailyUmsatzAmazon.db"
        self.dbpfad_itunes = "G:/Digitale Distribution/10_PLATTFORMEN/01_ITUNES/99_ABRECHNUNGEN/Datenbank/DailyUmsatziTunes.db"
        self.dbpfad_google = "G:/Digitale Distribution/10_PLATTFORMEN/15_GOOGLE/99_ABRECHNUNGEN/Datenbank/DailyUmsatzGoogle.db"
        self.dbpfad_ml = "G:/Digitale Distribution/10_PLATTFORMEN/16_AMAZON/99_Abrechnung/Daily Reports/DBManagement/Database/UmsatzforML.db"
        self.dbpfad_meta = 'G:/Digitale Distribution/10_PLATTFORMEN/01_ITUNES/50_ARTWORK_METADATEN/Datenbank/MasterMetadaten.db'
        self.localmode = False

    def get_localMode(self):
        return self.localmode

    def set_localMode(self, modus:bool):
        self.localmode = modus

    def getlocalpath(self):
        return self.pfadlokal

    def getTableAndDatefieldNamesForPF(self, pf) ->(str,str):
        """mögliche Inputs: amazon, google, itunes"""
        pfToTablename = {'amazon': ('Umsatz_Amazon', 'Datum'),
                         'itunes': ('Umsatz_iTunes', 'Datum'),
                         'google': ('UmsatzGoogle', 'Transaction_Date')}
        return pfToTablename[pf]

    def getConToDB(self, pf)->sqlite3.Connection:
        dbpfad = self.getDbPfadFuerPF(pf)
        return sqlite3.connect(dbpfad)

    def getDbPfadFuerPF(self, pf):
        #print('localmode: ' + str(self.localmode))
        if self.localmode == False:
            if pf == "amazon":
                dbpfad = self.dbpfad_amazon
            elif pf == "itunes":
                dbpfad = self.dbpfad_itunes
            elif pf == "google":
                dbpfad = self.dbpfad_google
            elif pf == "ml":
                dbpfad = self.dbpfad_ml
            else:
                dbpfad = None
        else:
            if pf == "ml":
                dbpfad = self.dbpfad_ml
            else:
                dbpfad = self.pfadlokal + "/" + pf +".db"
        return dbpfad

    def getTPdaten(self):
        """
        returns Dataframe with TPDaten for Movies
        :return: pd.Dataframe
        """
        conn = self.getConToDB('google')
        df = pd.read_sql('SELECT * FROM TPDaten;', con=conn)
        conn.close()
        return df

    def getTPTVdaten(self):
        """
        returns Dataframe with TVTPDaten for TV
        :return: pd.Dataframe
        """
        conn = self.getConToDB('google')
        df = pd.read_sql('SELECT * FROM TVTPDaten;', con=conn)
        conn.close()
        return df