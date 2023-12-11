# Speicherung von IQFeed-Daten:
# ------------------------------
# `store_iqfeed_data` ist für das Speichern von Tick-Daten. Tickdaten sind viel zu
# Speicherintensiv momentan, weswegen wir nur 1min-Frequenzdaten betrachten, die wir 
# normal über get_iqfeed_data(symbol, from, to, period='1min') herunterladen.



# ?QuantTools_settings
library(QuantTools)
QuantTools_settings_defaults()

# QuantTools_settings( settings = list(
#   iqfeed_host = 'localhost',
#   iqfeed_port = 'iqfeed port number'
# ) )

library(arrow)
# Compression example
path = "/home/beer/Downloads/frd_sample_stock_AAPL/AAPL_1min_sample.csv"
path_pqt = "/home/beer/Downloads/frd_sample_stock_AAPL/AAPL_1min_sample.parquet"
df <- read.csv(path)
df$timestamp <- as.POSIXct(df$timestamp,tz="EDT")
write_parquet(df, path_pqt, compression="gzip")

# Download Tickers
library(arrow)
#project_directory <- "G:\\Meine Ablage\\NewsTrading\\trading_bot"
project_directory <- "~/Documents/GithubProjekte/trading_bot"
ticker_name_mapper_path <- file.path(project_directory, "data_shared", "ticker_name_mapper_reduced.parquet")
tickers <- read_parquet(ticker_name_mapper_path)


from = '2022-12-03'
to   = '2023-10-04'
symbol = 'TRTN'

# Daily
stock <- get_iqfeed_data(symbol, from, to)

# Hourly
# get_iqfeed_data( symbol, from, to, period = 'hour' )

# Minutely
# get_iqfeed_data( symbol, from, to, period = 'minute' )

# Tick
# get_iqfeed_data( symbol, from, to = from, period = 'tick' )


# Store Market Data
symbols = c( 'AAPL')
path = paste( path.expand('~') , 'Market Data', 'iqfeed', sep = '/' )
start_date = '2022-12-01'

settings = list(
  iqfeed_storage = path,
  iqfeed_storage_from = start_date,
  iqfeed_symbols = symbols
)
QuantTools_settings( settings )

store_iqfeed_data()

# Get From Local Storage
aapl <- get_iqfeed_data( symbol = 'AAPL', from = '2023-10-02 15:00:00', to = '2023-10-03 09:33:00', period ='1min', local = T )

aapl <- aapl[aapl$time >= as.POSIXlt("2023-10-03 09:30:00", tz="UTC")]
# aapl <- aapl[aapl$trade_market_center == 19]
# aapl <- aapl[aapl$trade_conditions == 87]
aapl <- aapl[aapl$volume >= 5]
aapl_price <- na.omit(aapl[,.(time, open, high, low, close, volume)])

# Plot data
plot_ts( aapl_price )


############ Tick-Data and Conversion to OHLC ################
# aapl <- get_iqfeed_data( symbol = 'AAPL', from = '2023-10-02 15:00:00', to = '2023-10-03 09:33:00', period ='tick', local = T )
# aapl <- aapl[aapl$time >= as.POSIXlt("2023-10-03 09:00:00", tz="UTC")]
# 
# aapl <- aapl[aapl$trade_market_center == 19]
# aapl <- aapl[(aapl$trade_conditions == "01")]
# aapl <- aapl[aapl$volume >= 5]
# aapl_price <- na.omit(aapl[,.(time, price)])
# # Plot data
# plot_ts( aapl_price )
# 
# ohlc <- function(ttime,tprice,tvolume,fmt) {
#   ttime.int <- format(ttime,fmt)
#   data.frame(time = ttime[tapply(1:length(ttime),ttime.int,function(x) {head(x,1)})],
#              .Open = tapply(tprice,ttime.int,function(x) {head(x,1)}), 
#              .High = tapply(tprice,ttime.int,max),
#              .Low = tapply(tprice,ttime.int,min),
#              .Close = tapply(tprice,ttime.int,function(x) {tail(x,1)}),
#              .Volume = tapply(tvolume,ttime.int,function(x) {sum(x)}),
#              .Adjusted = tapply(tprice,ttime.int,function(x) {tail(x,1)}))
# } 
# ohlc(aapl$time, aapl$price, aapl$volume, fmt="%y-%m-%d %H:%M")


