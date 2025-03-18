import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import json
import ccxt.async_support as ccxt_async  # Renommé pour la version asynchrone
import ccxt  # Version synchrone pour le portfolio
import ta
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data(ttl=15)
def get_crypto_data(num_cryptos):
    async def fetch():
        """Récupère les données de CoinGecko de manière asynchrone"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "price_change_percentage": "1h,24h,7d,14d,30d,200d,1y",
                    "per_page": num_cryptos,
                    "page": 1,
                },
            ) as response:
                data = await response.json()
                df = pd.DataFrame(data)
                # Conversion des symboles en majuscules
                df["symbol"] = df["symbol"].str.upper()
                # Filtrage des stablecoins USD
                df = df[~df["symbol"].str.startswith("USD")]
                return df

    return asyncio.run(fetch())


async def fetch_single_ohlcv(exchange, symbol, timeframe, limit, base):
    """Récupère les données OHLCV pour une seule paire"""
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["symbol"] = base
        
        # Calcul des bandes de Bollinger
        df["bb_middle"] = ta.volatility.bollinger_mavg(df["close"], window=20)
        df["bb_upper"] = ta.volatility.bollinger_hband(df["close"], window=20)
        df["bb_lower"] = ta.volatility.bollinger_lband(df["close"], window=20)
        
        # Calcul de l'EMA 200 et de l'écart en pourcentage
        df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)
        df["ema_200_gap"] = ((df["close"] - df["ema_200"]) / df["ema_200"]) * 100
        
        # Vérification si le prix est dans les bandes
        df["in_bollinger_bands"] = (df["close"] <= df["bb_upper"]) & (df["close"] >= df["bb_lower"])
        
        return df
    except Exception as e:
        print(f"Erreur lors de la récupération des données pour {symbol}: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_ohlcv_for_cryptos(crypto_list, timeframe="1d", limit=200):
    async def fetch(crypto_list, timeframe="1d", limit=200):
        exchange = ccxt_async.bitmart({
            "enableRateLimit": True,
            "timeout": 30000,  # Augmentation du timeout
        })

        try:
            await exchange.load_markets()
            tasks = []

            for symbol, market in exchange.markets.items():
                if (
                    market.get("spot", False)
                    and market.get("quote") == "USDT"
                    and market.get("base") in crypto_list
                ):
                    tasks.append(
                        fetch_single_ohlcv(
                            exchange, symbol, timeframe, limit, market.get("base")
                        )
                    )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    print(f"Erreur lors de la récupération des données: {result}")
                    continue
                if result is not None:
                    try:
                        df = result
                        df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
                        df["return"] = df["close"].pct_change()
                        last_row = df.iloc[-1]
                        mean_return = df["return"].mean()
                        std_return = df["return"].std()
                        sharpe_ratio = (365**0.5) * (mean_return / std_return) if std_return != 0 else 0
                        valid_results.append(
                            last_row.to_dict()
                            | {
                                "sharpe_ratio": sharpe_ratio,
                                "mean_return": mean_return * 100,
                                "std_return": std_return * 100,
                                "in_bollinger_bands": last_row["in_bollinger_bands"],
                                "ema_200_gap": last_row["ema_200_gap"]
                            }
                        )
                    except Exception as e:
                        print(f"Erreur lors du traitement des données: {e}")
                        continue

            return pd.DataFrame(valid_results) if valid_results else pd.DataFrame()

        except Exception as e:
            print(f"Erreur générale: {e}")
            return pd.DataFrame()
        finally:
            await exchange.close()

    return asyncio.run(fetch(crypto_list, timeframe, limit))


def aggregate_crypto_data(df_coingecko, df_bitmart):
    """
    Agrège les données de CoinGecko et Bitmart en un seul DataFrame.

    Parameters:
        df_coingecko (DataFrame): Données de CoinGecko
        df_bitmart (DataFrame): Données de Bitmart

    Returns:
        DataFrame: DataFrame agrégé contenant les données des deux sources
    """
    # Sélection des colonnes pertinentes de CoinGecko
    df_coingecko_clean = df_coingecko[
        [
            "symbol",
            "name",
            "image",
            "market_cap_rank",
            "current_price",
            "market_cap",
            "total_volume",
            "ath",
            "ath_change_percentage",
            "price_change_percentage_24h_in_currency",
            "price_change_percentage_7d_in_currency",
            "price_change_percentage_14d_in_currency",
            "price_change_percentage_30d_in_currency",
            "price_change_percentage_200d_in_currency",
            "price_change_percentage_1y_in_currency",
        ]
    ].copy()

    # Conversion de la market cap en millions
    df_coingecko_clean["market_cap"] = df_coingecko_clean["market_cap"] / 1_000_000

    # Renommage des colonnes de Bitmart pour éviter les conflits
    df_bitmart_clean = df_bitmart[
        ["symbol", "close", "rsi_14", "sharpe_ratio", "mean_return", "std_return", "in_bollinger_bands", "ema_200_gap"]
    ].copy()

    # Conversion du booléen en texte pour les bandes de Bollinger
    df_bitmart_clean["in_bollinger_bands"] = df_bitmart_clean["in_bollinger_bands"].map({True: "Oui", False: "Non"})

    # Fusion des DataFrames sur la colonne symbol
    df_merged = pd.merge(df_coingecko_clean, df_bitmart_clean, on="symbol", how="right")
    df_merged = df_merged.sort_values(by="market_cap", ascending=False)

    return df_merged


def main():
    st.set_page_config(page_title="Crypto Dashboard", layout="wide")

    # Logo dans la barre latérale
    st.sidebar.image(
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTbih-00oWUiFEntl28hkDq-rHfqisKHJBhGg&s",
        width=100
    )

    # Barre latérale pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Sélectionnez une page:", ["Page 1", "Mon Portfolio"])

    # Création de deux colonnes pour les sélecteurs
    col1, col2 = st.columns([1, 2])

    # Sélection du nombre de cryptos
    with col1:
        num_cryptos = st.selectbox(
            "Nombre de cryptomonnaies à afficher",
            options=[("Top 10", 10), ("Top 50", 50), ("Top 100", 100), ("Top 200", 200)],
            format_func=lambda x: x[0],
            index=1
        )

    # Chargement initial des données
    with st.spinner(f"Chargement des {num_cryptos[0]} cryptomonnaies..."):
        df_coingecko_initial = get_crypto_data(num_cryptos[1])
        df_bitmart_initial = fetch_ohlcv_for_cryptos(df_coingecko_initial["symbol"].tolist())
        df_merged_initial = aggregate_crypto_data(df_coingecko_initial, df_bitmart_initial)
    
    # Sélection multiple pour exclure des cryptos
    with col2:
        cryptos_to_exclude = st.multiselect(
            "Exclure des cryptomonnaies (optionnel)",
            options=df_coingecko_initial['symbol'].tolist(),
            help="Sélectionnez les cryptomonnaies à exclure du tableau et des graphiques"
        )

    # Contenu principal
    st.title("Tableau de bord Crypto")

    # Filtrage des données sans recharger
    df_merged = df_merged_initial[~df_merged_initial['symbol'].isin(cryptos_to_exclude)].copy()

    # Configuration des colonnes pour les données fusionnées
    merged_config = {
        "image": st.column_config.ImageColumn("", help="Logo de la cryptomonnaie", pinned=True),
        "market_cap_rank": st.column_config.NumberColumn(
            "Rang", 
            help="Classement par capitalisation boursière",
            pinned=True,
            step=1
        ),
        "symbol": st.column_config.TextColumn("Symbole", pinned=True),
        "name": "Nom",
        "market_cap": st.column_config.NumberColumn(
            "Market Cap", 
            help="Capitalisation boursière en millions USD",
            format="$ %d M",
            step=1
        ),
        "current_price": st.column_config.NumberColumn(
            "Prix", 
            format="$ %.2f",
            step=0.01
        ),
        "ath": st.column_config.NumberColumn(
            "ATH", 
            format="$ %.2f",
            step=0.01
        ),
        "ath_change_percentage": st.column_config.NumberColumn(
            "Distance à l'ATH", format="%.2f%%"
        ),
        "rsi_14": st.column_config.NumberColumn("RSI 14", format="%.2f"),
        "in_bollinger_bands": st.column_config.TextColumn(
            "Dans les BB",
            help="Indique si le prix est à l'intérieur des bandes de Bollinger (20,2)"
        ),
        "ema_200_gap": st.column_config.NumberColumn(
            "Écart EMA 200", 
            format="%.2f%%",
            help="Écart en pourcentage entre le prix actuel et l'EMA 200"
        ),
        "sharpe_ratio": st.column_config.NumberColumn("Sharpe Ratio", format="%.2f"),
        "mean_return": st.column_config.NumberColumn(
            "Moyenne de retour", format="%.2f%%"
        ),
        "std_return": st.column_config.NumberColumn(
            "Écart-type de retour", format="%.2f%%"
        ),
        "price_change_percentage_24h_in_currency": st.column_config.NumberColumn(
            "Variation 24h", format="%.2f%%"
        ),
        "price_change_percentage_7d_in_currency": st.column_config.NumberColumn(
            "Variation 7j", format="%.2f%%"
        ),
        "price_change_percentage_14d_in_currency": st.column_config.NumberColumn(
            "Variation 14j", format="%.2f%%"
        ),
        "price_change_percentage_30d_in_currency": st.column_config.NumberColumn(
            "Variation 30j", format="%.2f%%"
        ),
        "price_change_percentage_200d_in_currency": st.column_config.NumberColumn(
            "Variation 200j", format="%.2f%%"
        ),
        "price_change_percentage_1y_in_currency": st.column_config.NumberColumn(
            "Variation 1an", format="%.2f%%"
        )
    }

    if page == "Mon Portfolio":
        st.title("Mon Portfolio Bitmart")
        
        # Configuration des clés API dans la sidebar
        with st.sidebar:
            st.subheader("Configuration API Bitmart")
            api_key = st.text_input("API Key", type="password")
            api_secret = st.text_input("API Secret", type="password")
            api_password = st.text_input("API Password", type="password")
            
        if api_key and api_secret and api_password:
            # Bouton pour charger les données
            if st.button("Charger mon portfolio"):
                try:
                    with st.spinner("Chargement de votre portfolio..."):
                        # Utilisation de ccxt synchrone
                        exchange = ccxt.bitmart({
                            'apiKey': api_key,
                            'secret': api_secret,
                            'uid': api_password,  # need uid for bitmart
                            'enableRateLimit': True
                        })
                        
                        # Récupération de la balance
                        balance = exchange.fetch_balance()
                        
                        # Filtrage des balances non nulles
                        non_zero_balances = {
                            currency: float(data['total'])
                            for currency, data in balance.items()
                            if isinstance(data, dict) and 
                            data.get('total', 0) is not None and 
                            float(data.get('total', 0)) > 0 and
                            currency not in ['total', 'used', 'free', 'info']
                        }
                        
                        if non_zero_balances:
                            # Création du pie chart
                            fig = go.Figure(data=[go.Pie(
                                labels=list(non_zero_balances.keys()),
                                values=list(non_zero_balances.values()),
                                hole=.3,
                                textinfo='label+percent'
                            )])
                            
                            # Personnalisation du layout
                            fig.update_layout(
                                title="Répartition de mon Portfolio",
                                template="plotly_dark",
                                showlegend=True,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            # Affichage du pie chart
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Affichage du tableau des balances
                            df_balance = pd.DataFrame(list(non_zero_balances.items()), columns=['Crypto', 'Quantité'])
                            st.dataframe(
                                df_balance,
                                column_config={
                                    "Crypto": st.column_config.TextColumn("Crypto", width="medium"),
                                    "Quantité": st.column_config.NumberColumn("Quantité", format="%.8f")
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                        else:
                            st.warning("Aucune balance non nulle trouvée dans votre compte.")
                            
                except Exception as e:
                    st.error(f"Erreur lors de la récupération des données: {str(e)}")
        else:
            st.info("Veuillez entrer vos clés API Bitmart dans la barre latérale pour voir votre portfolio.")

    if page == "Page 1":
        # Dictionnaire de correspondance pour les noms des colonnes
        column_names = {
            "market_cap_rank": "Rang",
            "current_price": "Prix",
            "market_cap": "Market Cap",
            "ath": "ATH",
            "ath_change_percentage": "Distance à l'ATH",
            "rsi_14": "RSI 14",
            "ema_200_gap": "Écart EMA 200",
            "sharpe_ratio": "Sharpe Ratio",
            "mean_return": "Moyenne de retour",
            "std_return": "Écart-type de retour",
            "price_change_percentage_24h_in_currency": "Variation 24h",
            "price_change_percentage_7d_in_currency": "Variation 7j",
            "price_change_percentage_14d_in_currency": "Variation 14j",
            "price_change_percentage_30d_in_currency": "Variation 30j",
            "price_change_percentage_200d_in_currency": "Variation 200j",
            "price_change_percentage_1y_in_currency": "Variation 1an"
        }

        # Application du style conditionnel pour plusieurs colonnes
        def color_bollinger(val):
            if val == "Oui":
                return 'background-color: #90EE90'
            elif val == "Non":
                return 'background-color: #FFB6C1'
            return ''

        def color_rsi(val):
            try:
                val = float(val)
                if val > 70:
                    return 'background-color: #90EE90'
                elif val < 30:
                    return 'background-color: #FFB6C1'
            except:
                pass
            return ''

        def color_percentage(val):
            try:
                val = float(val)
                if val > 0:
                    return 'background-color: #90EE90'
                elif val < 0:
                    return 'background-color: #FFB6C1'
            except:
                pass
            return ''

        # Application du style
        df_styled = df_merged.style\
            .map(color_bollinger, subset=['in_bollinger_bands'])\
            .map(color_rsi, subset=['rsi_14'])\
            .map(color_percentage, subset=[col for col in df_merged.columns if 'price_change_percentage' in col])\
            .map(color_percentage, subset=['ema_200_gap'])  # Même style que les variations de prix

        st.dataframe(
            df_styled,
            column_config=merged_config,
            column_order=list(merged_config.keys()),
            hide_index=True,
            use_container_width=True,
        )

        # Colonnes numériques disponibles pour les graphiques
        numeric_columns = df_merged.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()

        # Section pour le Scatter Plot
        st.subheader("Graphique de Dispersion")
        col1, col2 = st.columns(2)

        with col1:
            x_column = st.selectbox(
                "Choisir la variable X",
                options=numeric_columns,
                format_func=lambda x: column_names.get(x, x),
                index=(
                    numeric_columns.index("mean_return")
                    if "mean_return" in numeric_columns
                    else 0
                ),
            )

        with col2:
            y_column = st.selectbox(
                "Choisir la variable Y",
                options=numeric_columns,
                format_func=lambda x: column_names.get(x, x),
                index=(
                    numeric_columns.index("std_return")
                    if "std_return" in numeric_columns
                    else 0
                ),
            )

        # Création du scatter plot avec les noms d'affichage
        fig_scatter = px.scatter(
            df_merged,
            x=x_column,
            y=y_column,
            text="symbol",
            title=f"Relation entre {column_names.get(x_column, x_column)} et {column_names.get(y_column, y_column)}",
            template="plotly_dark",
            color="symbol",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hover_data=["name", x_column, y_column]
        )

        # Ajustement du texte au-dessus des points
        fig_scatter.update_traces(
            textposition="top center",
            marker=dict(size=12, line=dict(width=1, color='white'))
        )
        
        # Personnalisation du layout
        fig_scatter.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # Affichage du scatter plot
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Section pour le Bar Plot
        st.subheader("Graphique en Barres")

        y_column_bar = st.selectbox(
            "Choisir la variable à afficher",
            options=numeric_columns,
            format_func=lambda x: column_names.get(x, x),
            index=numeric_columns.index("rsi_14") if "rsi_14" in numeric_columns else 0,
        )

        # Création du bar plot avec le nom d'affichage
        fig_bar = px.bar(
            df_merged,
            x="symbol",
            y=y_column_bar,
            title=f"{column_names.get(y_column_bar, y_column_bar)} par Crypto",
            template="plotly_dark",
            text_auto=".2s",
            color=y_column_bar,
            color_continuous_scale="Viridis",
            hover_data=["name", y_column_bar]
        )

        # Personnalisation du bar plot
        fig_bar.update_traces(
            textposition="outside",
            textfont=dict(color="white"),
        )
        
        # Personnalisation du layout
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar_title="Valeur"
        )
        
        # Affichage du bar plot
        st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
