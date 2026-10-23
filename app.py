import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
import json

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(
    page_title="StudySound | НИУ ВШЭ",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA = Path("data")
TEMPO_BPM = {"медленный": 70, "средний": 100, "быстрый": 130}
DATA_FILE = Path("user_data.json")


# --- ЗАГРУЗКА МОДЕЛИ И ДАННЫХ ---
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA / "tracks.csv", sep=None, engine='python', on_bad_lines='skip', encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)
    embs = np.load(DATA / "embeddings.npy")
    feats = pd.read_csv(DATA / "audio_features.csv")
    return df.join(feats), embs


# --- СОХРАНЕНИЕ ПРЕДПОЧТЕНИЙ ---
def load_user_data():
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("liked_ids", [])), set(data.get("disliked_ids", []))
        except (json.JSONDecodeError, KeyError):
            pass
    return set(), set()


def save_user_data():
    data = {
        "liked_ids": list(st.session_state.liked_ids),
        "disliked_ids": list(st.session_state.disliked_ids)
    }
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --- ИНИЦИАЛИЗАЦИЯ ПАМЯТИ ---
if 'liked_ids' not in st.session_state:
    liked, disliked = load_user_data()
    st.session_state.liked_ids = liked
    st.session_state.disliked_ids = disliked
if 'results' not in st.session_state:
    st.session_state.results = None
if 'page' not in st.session_state:
    st.session_state.page = "Главная"

# ==========================================
# ЧЕРНО-БЕЛЫЙ СТИЛЬ (CSS)
# ==========================================
st.markdown("""
    <style>
        /* Глобальные настройки шрифта */
        html, body, [class*="css"] {
            font-family: 'Helvetica Neue', 'Arial', sans-serif;
        }
        /* Абсолютно черный фон */
        .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }
        /* Сайдбар в темно-сером цвете */
        [data-testid="stSidebar"] {
            background-color: #0A0A0A;
            border-right: 1px solid #222222;
            padding-top: 20px;
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important; 
        }
        /* Кнопки навигации в сайдбаре (как в приложении) */
        [data-testid="stSidebar"] .stButton>button {
            background-color: transparent;
            color: #AAAAAA !important;
            border: none;
            text-align: left;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: 500;
            transition: 0.2s;
        }
        [data-testid="stSidebar"] .stButton>button:hover {
            background-color: #222222;
            color: #FFFFFF !important;
            border-radius: 8px;
        }
        /* Заголовки */
        h1, h2, h3 {
            color: #FFFFFF !important;
            font-weight: 400;
            letter-spacing: 1px;
        }
        /* Карточки треков */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #111111;
            border: 1px solid #333333;
            border-radius: 12px;
            padding: 10px;
        }
        /* Кнопка подобрать музыку */
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border-radius: 30px;
            font-weight: bold;
            padding: 10px 30px;
        }
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #CCCCCC !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- САЙДБАР (НАВИГАЦИЯ) ---
with st.sidebar:
    # Логотип (если он есть, он будет белым, если нет - оставь заглушку)
    st.markdown("<h2 style='text-align: center;'>🎧 StudySound</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 12px; color: #666;'>При поддержке НИУ ВШЭ</p>",
                unsafe_allow_html=True)
    st.divider()

    # Меню навигации
    if st.button("🏠 Главная", use_container_width=True):
        st.session_state.page = "Главная"
        st.rerun()
    if st.button("📊 Ваш вкус", use_container_width=True):
        st.session_state.page = "Вкус"
        st.rerun()
    if st.button(f"❤️ Плейлист ({len(st.session_state.liked_ids)})", use_container_width=True):
        st.session_state.page = "Плейлист"
        st.rerun()

    st.divider()
    if st.button("🗑️ Очистить данные", use_container_width=True):
        st.session_state.liked_ids = set()
        st.session_state.disliked_ids = set()
        st.session_state.results = None
        save_user_data()
        st.rerun()

# --- АНИМАЦИЯ ЗАГРУЗКИ ---
loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown("""
        <div style="text-align: center; margin-top: 20%;">
            <div style="border: 4px solid #333; border-top: 4px solid #FFF; border-radius: 50%; width: 60px; height: 60px; animation: spin 1s linear infinite; margin: 0 auto;"></div>
            <p style="color: #FFF; margin-top: 20px; font-weight: 300;">Загрузка нейросети...</p>
        </div>
        <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
    """, unsafe_allow_html=True)

try:
    model = load_model()
    df, doc_embs = load_data()
except Exception as e:
    st.error(f"Ошибка загрузки: {e}")
    st.stop()

loading_placeholder.empty()

# ==========================================
# СТРАНИЦА 1: ГЛАВНАЯ (ПОИСК)
# ==========================================
if st.session_state.page == "Главная":
    st.title("StudySound: Подберём музыку под твою задачу")
    st.write("Выбери параметры, и нейросеть найдет идеальный трек.")
    st.write("---")

    # Быстрые кнопки (как на фото)
    st.write("**Быстрый выбор:**")
    cols = st.columns(4)
    if cols[0].button("📚 Учёба"):
        task, energy, vocal, tempo, mood, genre = "учёба", 3, "no", "средний", "спокойное", "любой"
        st.session_state.quick_search = True
    if cols[1].button("🏃 Спорт"):
        task, energy, vocal, tempo, mood, genre = "спорт", 5, "no", "быстрый", "бодрое", "любой"
        st.session_state.quick_search = True
    if cols[2].button("🧘 Расслабление"):
        task, energy, vocal, tempo, mood, genre = "расслабление", 1, "no", "медленный", "спокойное", "любой"
        st.session_state.quick_search = True
    if cols[3].button("💻 Работа"):
        task, energy, vocal, tempo, mood, genre = "работа", 3, "no", "средний", "нейтральное", "любой"
        st.session_state.quick_search = True

    with st.form("profile"):
        c1, c2, c3 = st.columns(3)
        with c1:
            task = st.selectbox("Задача", ["учёба", "работа", "спорт", "чтение", "расслабление"],
                                index=["учёба", "работа", "спорт", "чтение", "расслабление"].index(
                                    task) if 'task' in locals() else 0)
            energy = st.slider("Энергия (1–5)", 1, 5, energy if 'energy' in locals() else 3)
        with c2:
            vocal = st.radio("Вокал", ["no", "yes"], format_func=lambda x: "без вокала" if x == "no" else "можно",
                             index=0 if ('vocal' not in locals() or vocal == "no") else 1)
            tempo = st.select_slider("Темп", ["медленный", "средний", "быстрый"],
                                     value=tempo if 'tempo' in locals() else "средний")
        with c3:
            mood = st.selectbox("Настроение", ["спокойное", "нейтральное", "бодрое", "грустное"],
                                index=["спокойное", "нейтральное", "бодрое", "грустное"].index(
                                    mood) if 'mood' in locals() else 0)
            genre = st.selectbox("Жанр", ["любой", "lo-fi", "ambient", "classical", "electronic", "rap"])
        submitted = st.form_submit_button("Подобрать музыку")

    if submitted or st.session_state.get('quick_search', False):
        st.session_state.quick_search = False
        query = f"Задача: {task}. Энергия: {energy}/5. Вокал: {vocal}. Темп: {tempo}. Настроение: {mood}. Жанр: {genre}."
        q_emb = model.encode(query, normalize_embeddings=True)

        if st.session_state.liked_ids:
            liked_indices = [df.index[df['track_id'].astype(str) == str(tid)][0] for tid in st.session_state.liked_ids
                             if not df[df['track_id'].astype(str) == str(tid)].empty]
            if liked_indices:
                liked_embs = doc_embs[liked_indices]
                liked_profile = np.mean(liked_embs, axis=0)
                q_emb = 0.8 * q_emb + 0.2 * liked_profile
                q_emb = q_emb / np.linalg.norm(q_emb)

        hits = semantic_search(q_emb, doc_embs, top_k=min(len(df), 50))[0]

        results = []
        for h in hits:
            row = df.iloc[h["corpus_id"]]
            text_for_task = str(row.get('suitability', '')) + str(row.get('description_ru', ''))
            task_match = 1.0 if str(task) in text_for_task else 0.0
            penalty = 0.6 if (vocal == "no" and row["vocal"] == "yes") else 0.0

            if str(row['track_id']) in st.session_state.disliked_ids:
                continue

            e_sim = 1 - abs(row["energy_1_5"] - energy) / 4
            b_sim = max(0.0, 1 - abs(row["bpm"] - TEMPO_BPM[tempo]) / 50)
            audio_sim = (0.6 * e_sim + 0.4 * b_sim)
            final = (0.5 * h["score"]) + (0.5 * audio_sim) + (0.2 * task_match) - penalty

            if str(row['track_id']) in st.session_state.liked_ids:
                final += 0.1
            results.append((final, row))

        results.sort(key=lambda x: x[0], reverse=True)
        unique_results, seen_ids = [], set()
        for score, row in results:
            if row['track_id'] not in seen_ids:
                unique_results.append((score, row))
                seen_ids.add(row['track_id'])
            if len(unique_results) == 5:
                break
        st.session_state.results = unique_results

    if st.session_state.results:
        st.subheader("Твой микс прямо сейчас")
        for score, row in st.session_state.results[:5]:
            with st.container(border=True):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"### {row['title']} — {row['artist']}")
                    st.caption(
                        f"Match: {score:.2f} · {row['genre']} · {row['bpm']:.0f} BPM · энергия {row['energy_1_5']}/5")

                    vals = [row["energy_1_5"] / 5, min(row["bpm"] / 160, 1), min(row["centroid"] / 4000, 1)]
                    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=["Энергия", "Темп", "Яркость", "Энергия"],
                                                    fill="toself", line_color="#FFFFFF",
                                                    fillcolor="rgba(255,255,255,0.2)"))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), showlegend=False,
                                      height=200, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{row['track_id']}")

                with col2:
                    audio_path = DATA / row["file_path"]
                    if audio_path.exists():
                        st.audio(str(audio_path))
                    else:
                        st.warning("Аудиофайл не найден")

                    btn_col1, btn_col2 = st.columns(2)
                    is_liked = str(row['track_id']) in st.session_state.liked_ids
                    is_disliked = str(row['track_id']) in st.session_state.disliked_ids

                    if btn_col1.button("❤️" if is_liked else "🤍", key=f"like_{row['track_id']}",
                                       use_container_width=True):
                        if is_liked:
                            st.session_state.liked_ids.remove(str(row['track_id']))
                        else:
                            st.session_state.liked_ids.add(str(row['track_id']))
                            st.session_state.disliked_ids.discard(str(row['track_id']))
                        save_user_data()
                        st.rerun()
                    if btn_col2.button("❌" if is_disliked else "👎", key=f"dislike_{row['track_id']}",
                                       use_container_width=True):
                        if is_disliked:
                            st.session_state.disliked_ids.remove(str(row['track_id']))
                        else:
                            st.session_state.disliked_ids.add(str(row['track_id']))
                            st.session_state.liked_ids.discard(str(row['track_id']))
                        save_user_data()
                        st.rerun()

# ==========================================
# СТРАНИЦА 2: ВАШ ВКУС (АНАЛИТИКА)
# ==========================================
elif st.session_state.page == "Вкус":
    st.title("📊 Мой вкус")
    st.write("Здесь нейросеть анализирует ваши предпочтения на основе лайкнутых треков.")
    st.write("---")

    if not st.session_state.liked_ids:
        st.info("Вы пока не лайкнули ни одного трека. Перейдите на Главную и добавьте музыку в плейлист.")
    else:
        liked_df = df[df['track_id'].astype(str).isin(st.session_state.liked_ids)]

        c1, c2, c3 = st.columns(3)
        c1.metric("Треков в плейлисте", len(liked_df))
        c2.metric("Средний BPM", f"{liked_df['bpm'].mean():.0f}")
        c3.metric("Средняя энергия", f"{liked_df['energy_1_5'].mean():.1f} / 5")

        st.write("---")
        st.subheader("Ваши любимые жанры")
        genre_counts = liked_df['genre'].value_counts()
        st.bar_chart(genre_counts, color="#FFFFFF")

# ==========================================
# СТРАНИЦА 3: ПЛЕЙЛИСТ
# ==========================================
elif st.session_state.page == "Плейлист":
    st.title("❤️ Мой плейлист")
    st.write("---")

    if not st.session_state.liked_ids:
        st.info("Здесь пока пусто. Перейдите на Главную и нажмите 🤍 на треках, которые вам понравились.")
    else:
        liked_df = df[df['track_id'].astype(str).isin(st.session_state.liked_ids)]

        for _, row in liked_df.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{row['title']}** — {row['artist']}")
                    st.caption(f"{row['genre']} · {row['bpm']:.0f} BPM · энергия {row['energy_1_5']}/5")
                with col2:
                    audio_path = DATA / row["file_path"]
                    if audio_path.exists():
                        st.audio(str(audio_path))
                    if st.button("🗑️ Удалить", key=f"remove_{row['track_id']}", use_container_width=True):
                        st.session_state.liked_ids.remove(str(row['track_id']))
                        save_user_data()
                        st.rerun()