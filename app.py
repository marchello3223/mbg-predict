from flask import Flask, request, jsonify, render_template
from pathlib import Path
import pandas as pd
import joblib


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# PATH PROJECT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "machinelearning"
WEBSCRAPER_DIR = BASE_DIR / "webscraper"


# ============================================================
# MODEL
# ============================================================

MODEL_PATHS = {
    "random_forest":
        MODEL_DIR / "random_forest_model.pkl",

    "decision_tree":
        MODEL_DIR / "decision_tree_model.pkl",

    "xgboost":
        MODEL_DIR / "xgboost_model.pkl"
}


# ============================================================
# NAMA MODEL
# ============================================================

MODEL_NAMES = {
    "random_forest":
        "Random Forest Regression",

    "decision_tree":
        "Decision Tree Regression",

    "xgboost":
        "XGBoost Regression"
}


# ============================================================
# LOAD MODEL
# ============================================================

models = {}

for model_key, model_path in MODEL_PATHS.items():

    if model_path.exists():

        models[model_key] = joblib.load(model_path)


# ============================================================
# HOME / DASHBOARD
# ============================================================

@app.route("/")
def home():

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    clean_path = (
        WEBSCRAPER_DIR /
        "mbg_dataset_clean.csv"
    )

    ml_path = (
        WEBSCRAPER_DIR /
        "mbg_dataset_ml.csv"
    )

    comparison_path = (
        MODEL_DIR /
        "hasil_perbandingan_model.csv"
    )


    # --------------------------------------------------------
    # DATASET CLEAN
    # --------------------------------------------------------

    if clean_path.exists():

        clean_df = pd.read_csv(
            clean_path,
            sep=";"
        )

    else:

        clean_df = pd.DataFrame()


    # --------------------------------------------------------
    # DATASET ML
    # --------------------------------------------------------

    if ml_path.exists():

        ml_df = pd.read_csv(
            ml_path,
            sep=";"
        )

    else:

        ml_df = pd.DataFrame()


    # --------------------------------------------------------
    # PERBANDINGAN MODEL
    # --------------------------------------------------------

    if comparison_path.exists():

        comparison_df = pd.read_csv(
            comparison_path,
            sep=";"
        )

    else:

        comparison_df = pd.DataFrame()


    # ========================================================
    # STATISTIK DATA
    # ========================================================

    jumlah_data_clean = len(clean_df)

    jumlah_data_ml = len(ml_df)


    # --------------------------------------------------------
    # JUMLAH PROVINSI
    # --------------------------------------------------------

    if (
        not ml_df.empty
        and "provinsi" in ml_df.columns
    ):

        jumlah_provinsi = (
            ml_df["provinsi"]
            .nunique()
        )

    elif (
        not clean_df.empty
        and "provinsi" in clean_df.columns
    ):

        jumlah_provinsi = (
            clean_df["provinsi"]
            .nunique()
        )

    else:

        jumlah_provinsi = 0


    # --------------------------------------------------------
    # JUMLAH MODEL
    # --------------------------------------------------------

    jumlah_model = len(models)


    # ========================================================
    # RENTANG TAHUN
    # ========================================================

    tahun_awal = "-"
    tahun_akhir = "-"


    # --------------------------------------------------------
    # PRIORITAS: DATA ML
    # Karena mbg_dataset_ml.csv punya kolom "tahun"
    # --------------------------------------------------------

    if (
        not ml_df.empty
        and "tahun" in ml_df.columns
    ):

        tahun_data = pd.to_numeric(
            ml_df["tahun"],
            errors="coerce"
        ).dropna()


        if not tahun_data.empty:

            tahun_awal = int(
                tahun_data.min()
            )

            tahun_akhir = int(
                tahun_data.max()
            )


    # --------------------------------------------------------
    # FALLBACK: DATA CLEAN
    # Kalau dataset ML tidak punya tahun
    # --------------------------------------------------------

    elif (
        not clean_df.empty
        and "tanggal" in clean_df.columns
    ):

        tanggal_data = pd.to_datetime(
            clean_df["tanggal"],
            errors="coerce"
        ).dropna()


        if not tanggal_data.empty:

            tahun_awal = int(
                tanggal_data.dt.year.min()
            )

            tahun_akhir = int(
                tanggal_data.dt.year.max()
            )


    # ========================================================
    # KIRIM DATA KE DASHBOARD
    # ========================================================

    return render_template(
        "dashboard.html",

        jumlah_data_clean=
            jumlah_data_clean,

        jumlah_data_ml=
            jumlah_data_ml,

        jumlah_provinsi=
            jumlah_provinsi,

        jumlah_model=
            jumlah_model,

        tahun_awal=
            tahun_awal,

        tahun_akhir=
            tahun_akhir,

        comparison_data=
            comparison_df.to_dict(
                orient="records"
            )
    )


# ============================================================
# HALAMAN PREDIKSI
# ============================================================

@app.route("/prediksi")
def prediksi():

    return render_template(
        "index.html"
    )


# ============================================================
# HALAMAN PERBANDINGAN MODEL
# ============================================================

@app.route("/perbandingan-model")
def perbandingan_model():

    comparison_path = (
        MODEL_DIR /
        "hasil_perbandingan_model.csv"
    )


    if comparison_path.exists():

        comparison_df = pd.read_csv(
            comparison_path,
            sep=";"
        )

    else:

        comparison_df = pd.DataFrame()


    return render_template(
        "comparison.html",

        comparison_data=
            comparison_df.to_dict(
                orient="records"
            )
    )


# ============================================================
# HALAMAN DATA
# ============================================================

@app.route("/data")
def data_page():

    # --------------------------------------------------------
    # FILE DATA
    # --------------------------------------------------------

    clean_path = WEBSCRAPER_DIR / "mbg_dataset_clean.csv"

    ml_path = WEBSCRAPER_DIR / "mbg_dataset_ml.csv"


    # --------------------------------------------------------
    # BACA DATA CLEAN
    # --------------------------------------------------------

    clean_df = pd.read_csv(
        clean_path,
        sep=";"
    )


    # --------------------------------------------------------
    # BACA DATA ML
    # --------------------------------------------------------

    ml_df = pd.read_csv(
        ml_path,
        sep=";"
    )


    # --------------------------------------------------------
    # NORMALISASI NAMA KOLOM
    # --------------------------------------------------------

    clean_df = clean_df.rename(
        columns={
            "tanggal_kasus": "tanggal",
            "sumber_asal": "sumber"
        }
    )


    # --------------------------------------------------------
    # DATA YANG DITAMPILKAN
    # --------------------------------------------------------

    data_df = clean_df.copy()


    # --------------------------------------------------------
    # FORMAT TANGGAL
    # --------------------------------------------------------

    if "tanggal" in data_df.columns:

        data_df["tanggal"] = pd.to_datetime(
            data_df["tanggal"],
            errors="coerce"
        ).dt.strftime(
            "%Y-%m-%d"
        )


    # --------------------------------------------------------
    # UBAH NaN MENJADI KOSONG
    # --------------------------------------------------------

    data_df = data_df.fillna("")


    # --------------------------------------------------------
    # DATA DIUBAH MENJADI DICTIONARY
    # --------------------------------------------------------

    data = data_df.to_dict(
        orient="records"
    )


    # --------------------------------------------------------
    # DAFTAR PROVINSI
    # --------------------------------------------------------

    provinsi_list = sorted(
        ml_df["provinsi"]
        .dropna()
        .unique()
        .tolist()
    )


    # --------------------------------------------------------
    # STATISTIK DATA
    # --------------------------------------------------------

    total_data = len(clean_df)

    total_ml = len(ml_df)

    total_provinsi = len(provinsi_list)


    # --------------------------------------------------------
    # RENDER HALAMAN
    # --------------------------------------------------------

    return render_template(
        "data.html",

        data=data,

        total_data=total_data,

        total_ml=total_ml,

        total_provinsi=total_provinsi,

        provinsi_list=provinsi_list
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()


        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        tahun = int(
            data.get("tahun")
        )

        bulan = int(
            data.get("bulan")
        )

        provinsi = data.get(
            "provinsi"
        )

        selected_model = data.get(
            "model",
            "random_forest"
        )


        # ----------------------------------------------------
        # CEK TAHUN
        # ----------------------------------------------------

        if tahun < 2025:

            return jsonify({

                "status":
                    "no_data",

                "message":
                    (
                        "Data MBG yang digunakan "
                        "dalam sistem ini belum "
                        f"tersedia untuk tahun {tahun}."
                    )

            })


        # ----------------------------------------------------
        # CEK MODEL
        # ----------------------------------------------------

        if selected_model not in models:

            return jsonify({

                "status":
                    "error",

                "error":
                    "Model yang dipilih tidak tersedia."

            }), 400


        # ----------------------------------------------------
        # CEK PROVINSI
        # ----------------------------------------------------

        valid_provinces = [

            "DKI Jakarta",

            "Jawa Barat",

            "Jawa Tengah",

            "Jawa Timur",

            "Kalimantan Selatan",

            "Nusa Tenggara Timur",

            "Sulawesi Tengah"

        ]


        if provinsi not in valid_provinces:

            return jsonify({

                "status":
                    "error",

                "error":
                    "Provinsi tidak tersedia dalam dataset."

            }), 400


        # ----------------------------------------------------
        # LOAD DATASET ML
        # ----------------------------------------------------

        ml_path = (
            WEBSCRAPER_DIR /
            "mbg_dataset_ml.csv"
        )

        if ml_path.exists():

            ml_df = pd.read_csv(
                ml_path,
                sep=";"
            )

        else:

            ml_df = pd.DataFrame()


        # ----------------------------------------------------
        # DATA HISTORIS PROVINSI
        # ----------------------------------------------------

        if (
            not ml_df.empty
            and "provinsi" in ml_df.columns
        ):

            jumlah_data_provinsi = len(
                ml_df[
                    ml_df["provinsi"] == provinsi
                ]
            )

            total_data_ml = len(
                ml_df
            )

        else:

            jumlah_data_provinsi = 0

            total_data_ml = 0


        # ----------------------------------------------------
        # STATUS DATA HISTORIS
        # ----------------------------------------------------

        if jumlah_data_provinsi == 0:

            status_data = (
                "Belum ada data historis"
            )

        elif jumlah_data_provinsi <= 3:

            status_data = (
                "Data historis terbatas"
            )

        else:

            status_data = (
                "Data historis tersedia"
            )


        # ----------------------------------------------------
        # MEMBUAT INPUT MODEL
        # ----------------------------------------------------

        input_data = {

            "tahun":
                tahun,

            "bulan":
                bulan

        }


        # ----------------------------------------------------
        # ONE-HOT PROVINSI
        # ----------------------------------------------------

        for province in valid_provinces:

            column_name = (
                "provinsi_"
                + province
            )

            input_data[column_name] = (
                1
                if province == provinsi
                else 0
            )


        # ----------------------------------------------------
        # DATAFRAME
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )


        # ----------------------------------------------------
        # URUTAN KOLOM
        # ----------------------------------------------------

        feature_columns = [

            "tahun",

            "bulan",

            "provinsi_DKI Jakarta",

            "provinsi_Jawa Barat",

            "provinsi_Jawa Tengah",

            "provinsi_Jawa Timur",

            "provinsi_Kalimantan Selatan",

            "provinsi_Nusa Tenggara Timur",

            "provinsi_Sulawesi Tengah"

        ]


        input_df = input_df[
            feature_columns
        ]


        # ----------------------------------------------------
        # PREDIKSI
        # ----------------------------------------------------

        model = models[
            selected_model
        ]


        prediction = model.predict(
            input_df
        )[0]


        # ----------------------------------------------------
        # BULATKAN HASIL
        # ----------------------------------------------------

        prediction = round(
            prediction
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "status":
                "success",

            "prediksi_jumlah_kasus":
                prediction,

            "model":
                selected_model,

            "model_name":
                MODEL_NAMES[
                    selected_model
                ],

            "tahun":
                tahun,

            "bulan":
                bulan,

            "provinsi":
                provinsi,

            "data_historis_provinsi":
                jumlah_data_provinsi,

            "total_data_ml":
                total_data_ml,

            "status_data":
                status_data

        })


    except Exception as e:

        return jsonify({

            "status":
                "error",

            "error":
                str(e)

        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )