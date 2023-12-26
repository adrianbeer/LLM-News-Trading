# Speicherung von IQFeed-Daten:
# ------------------------------
# `store_iqfeed_data` ist f?r das Speichern von Tick-Daten. Tickdaten sind viel zu
# Speicherintensiv momentan, weswegen wir nur 1min-Frequenzdaten betrachten, die wir 
# normal ?ber get_iqfeed_data(symbol, from, to, period='1min') herunterladen.

# Special Installation Instruction
# - Arrow/Parquet:
# install.packages("arrow")
# - tzdb wird ben?tigt f?r Zeitzonen von arrow:
# install.packages("tzdb")

library(quantmod)
getSplits("RELIANCE.NS")

library(here)
library(yaml) 
config <- yaml.load_file(here("src/config.yaml"))

# ?QuantTools_settings
library(QuantTools)
QuantTools_settings_defaults()

# QuantTools_settings( settings = list(
#   iqfeed_host = 'localhost',
#   iqfeed_port = 'iqfeed port number'
# ) )

library(arrow)

# Download Tickers
ticker_name_mapper_path <- here("data_shared", "ticker_name_mapper_reduced.parquet")
tickers <- read_parquet(ticker_name_mapper_path)


# Requires minimal changes to download the daily time series...
from = '2010-01-01'
to   = '2023-12-15'
N <- length(tickers$stocks)
stopped_at = 1591

for (symbol in c(tickers$stocks[stopped_at:N], "SPY")) {
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
  
  path_pqt = file.path(config$data$iqfeed$minute$raw, paste(symbol, "_1min", ".parquet", sep=""))
  # Check "<R_HOME>/share/zoneinfo/zone.tab" for more time zones/info
  # df$time <- as.POSIXct(df$timestamp,tz="America/New_York")
  write_parquet(stock_1min, path_pqt, compression="gzip")
}

# system('shutdown -s')

