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
aapl <- get_iqfeed_data( symbol = 'AAPL', from = '2023-10-02 15:00:00', to = '2023-10-03 09:33:00', period = 'minute', local = T )

aapl <- aapl[aapl$time >= as.POSIXlt("2023-10-02 15:58:00", tz="UTC")]
aapl <- aapl[aapl$trade_market_center == 19]
aapl <- aapl[aapl$trade_conditions == 87]
aapl <- aapl[aapl$volume >= 5]
aapl_price <- na.omit(aapl[,.(time,price)])

# Plot data
plot_ts( aapl_price )






