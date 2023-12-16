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


from = '2010-01-01'
to   = '2023-12-15'
N <- length(tickers$stocks)

for (symbol in tickers$stocks[708:N]) {
  print(symbol)
  skip_to_next <- FALSE
  # stock_1min <- get_iqfeed_data( symbol, from, to, period = 'minute')
  tryCatch( 
    { stock_1min <- get_iqfeed_data( symbol, from, to, period = 'minute') }, 
    error = function(e) {
      print(e)
      skip_to_next <- TRUE
    }
  )
  if (skip_to_next|is.null(stock_1min)) { next }
  
  path_pqt = file.path("D:", "IQFeedData", paste(symbol, "_1min", ".parquet", sep=""))
  # Check "<R_HOME>/share/zoneinfo/zone.tab" for more time zones/info
  # df$time <- as.POSIXct(df$timestamp,tz="America/New_York")
  write_parquet(stock_1min, path_pqt, compression="gzip")
}

# system('shutdown -s')

