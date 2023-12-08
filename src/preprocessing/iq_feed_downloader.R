# ?QuantTools_settings
library(QuantTools)
QuantTools_settings_defaults()

# QuantTools_settings( settings = list(
#   iqfeed_host = 'localhost',
#   iqfeed_port = 'iqfeed port number'
# ) )


library(arrow)
project_directory <- "G:\\Meine Ablage\\NewsTrading\\trading_bot"
ticker_name_mapper <- paste(project_directory, "\\data_shared\\ticker_name_mapper_reduced.parquet", sep="")
tickers <- read_parquet(ticker_name_mapper)


from = '2022-12-03'
to   = '2023-10-04'
symbol = 'TRTN'

# Daily
stock <- get_iqfeed_data(symbol, from, to)

# Hourly
# get_iqfeed_data( symbol, from, to, period = 'hour' )

# Tick
# get_iqfeed_data( symbol, from, to = from, period = 'tick' )


# Store Market Data
symbols = c( 'AAPL','IBM','PG','WFC')
path = paste( path.expand('~') , 'Market Data', 'iqfeed', sep = '/' )
start_date = '2023-10-01'

settings = list(
  iqfeed_storage = path,
  iqfeed_storage_from = start_date,
  iqfeed_symbols = symbols
)
QuantTools_settings( settings )

store_iqfeed_data()

# Get From Local Storage
aapl <- get_iqfeed_data( symbol = 'AAPL', from = '2023-10-02 15:00:00', to = '2023-10-03 09:33:00', period = 'tick', local = T )

aapl <- aapl[aapl$time >= as.POSIXlt("2023-10-02 15:58:00", tz="UTC")]
aapl <- aapl[aapl$trade_market_center == 19]
aapl <- aapl[aapl$trade_conditions == 87]
aapl <- aapl[aapl$volume >= 5]
aapl_price <- na.omit(aapl[,.(time,price)])

# Plot data
plot_ts( aapl_price )






