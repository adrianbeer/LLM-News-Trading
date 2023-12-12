# Speicherung von IQFeed-Daten:
# ------------------------------
# `store_iqfeed_data` ist für das Speichern von Tick-Daten. Tickdaten sind viel zu
# Speicherintensiv momentan, weswegen wir nur 1min-Frequenzdaten betrachten, die wir 
# normal über get_iqfeed_data(symbol, from, to, period='1min') herunterladen.

# Installation
# - Arrow/Parquet:
# install.packages("arrow")
# - tzdb wird benötigt für Zeitzonen von arrow:
# install.packages("tzdb")

# ?QuantTools_settings
library(QuantTools)
QuantTools_settings_defaults()

# QuantTools_settings( settings = list(
#   iqfeed_host = 'localhost',
#   iqfeed_port = 'iqfeed port number'
# ) )

library(arrow)

# Download Tickers
project_directory <- "G:\\Meine Ablage\\NewsTrading\\trading_bot"
# project_directory <- "~/Documents/GithubProjekte/trading_bot"
ticker_name_mapper_path <- file.path(project_directory, "data_shared", "ticker_name_mapper_reduced.parquet")
tickers <- read_parquet(ticker_name_mapper_path)


from = '2022-01-01'
to   = '2023-01-01'
symbol = 'TRTN'

# Daily
stock <- get_iqfeed_data(symbol, from, to)

# Minutely
stock_1min <- get_iqfeed_data( symbol, from, to, period = 'minute')

# Compression example
path = "C:/Users/Adria/Downloads/frd_stock_sample/TSLA_1min_sample.csv"
path_pqt = "C:/Users/Adria/Downloads/frd_stock_sample/TSLA_1day_sample.parquet"
df <- read.csv(path)
# Check "<R_HOME>/share/zoneinfo/zone.tab" for more time zones/info
df$timestamp <- as.POSIXct(df$timestamp,tz="America/New_York")
write_parquet(df, path_pqt, compression="gzip")


