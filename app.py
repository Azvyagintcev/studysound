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

        /* Сайдбар в темно-сером цвете с белой границей */
        [data-testid="stSidebar"] {
            background-image: linear-gradient(180deg, #121212 0%, #000000 100%);
            border-right: 1px solid #333333;
        }

        /* Белый текст в сайдбаре */
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important; 
        }

        /* Аккордеоны в сайдбаре */
        [data-testid="stSidebar"] .streamlit-expanderHeader {
            background-color: #222222;
            border-radius: 8px;
            color: #FFFFFF;
            border: 1px solid #444;
        }

        /* Заголовки с тонкой белой линией снизу */
        h1, h2, h3 {
            color: #FFFFFF !important;
            border-bottom: 1px solid #444444;
            padding-bottom: 10px;
            font-weight: 300;
        }

        /* Белые кнопки с черным текстом */
        .stButton>button {
            background-color: #FFFFFF;
            color: #000000;
            border: 1px solid #FFFFFF;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #CCCCCC;
            color: #000000;
            border: 1px solid #CCCCCC;
        }

        /* Кнопка отправки формы */
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: none;
            font-weight: bold;
            transition: 0.3s;
        }
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #CCCCCC !important;
        }

        /* Карточки треков (темно-серые с тонкой рамкой) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #111111;
            border: 1px solid #333333;
            border-radius: 12px;
        }

        /* Белые метрики */
        [data-testid="stMetricValue"] {
            color: #FFFFFF;
        }
    </style>
""", unsafe_allow_html=True)

# --- КОНТЕНТ САЙДБАРА ---
with st.sidebar:
    # Добавляем логотип НИУ ВШЭ (используем официальный логотип с сайта)
    st.image("hse_logo.png", width=120)
    st.markdown("### StudySound Recommender")
    st.caption("Проект при поддержке НИУ ВШЭ")

    st.divider()
    with st.expander("ℹ️ О проекте", expanded=True):
        st.write(
            "MVP-приложение, использующее нейросети для семантического поиска музыки под ваши задачи. Разработано в рамках выездной сессии НИУ ВШЭ.")
    with st.expander("🛠 Технологии"):
        st.markdown("- **Streamlit** (UI)\n- **Sentence-Transformers** (ИИ)\n- **Librosa** (Аудио)")
    with st.expander("👥 Команда"):
        st.markdown("- **HoantHelio** — IT / Backend\n- **DJFenix** — Музыкальный куратор")
    st.divider()

    if st.button("🗑️ Очистить лайки и дизлайки", use_container_width=True):
        st.session_state.liked_ids = set()
        st.session_state.disliked_ids = set()
        st.session_state.results = None
        save_user_data()
        st.rerun()

# --- ОСНОВНОЙ ЭКРАН ---
st.title("🎧 StudySound Recommender")
st.caption("Подберём фоновую музыку под вашу задачу и состояние")

# --- АНИМАЦИЯ ЗАГРУЗКИ ---
loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown("""
        <style>
        /* Крутящийся лоадер в цветах ВШЭ */
        .hse-loader {
            border: 8px solid #E6F0FF; /* Светло-голубой фон */
            border-top: 8px solid #0050CF; /* Фирменный синий ВШЭ */
            border-radius: 50%;
            width: 80px;
            height: 80px;
            animation: spin 1s linear infinite;
            margin: 15% auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .hse-loader-text {
            text-align: center;
            color: #0050CF;
            font-family: 'Arial', sans-serif;
            font-size: 22px;
            font-weight: bold;
            margin-top: 20px;
        }
        </style>
        <div class="hse-loader"></div>
        <div class="hse-loader-text">Нейросеть просыпается... Подгружаем треки...</div>
    """, unsafe_allow_html=True)

# Небольшая пауза, чтобы пользователь успел увидеть анимацию даже на быстрых запусках
time.sleep(1.5)

# --- ЗАГРУЗКА МОДЕЛИ И ДАННЫХ ---
try:
    model = load_model()
    df, doc_embs = load_data()
except Exception as e:
    st.error(f"Ошибка загрузки: {e}")
    st.stop()

# Убираем анимацию, так как всё загрузилось
loading_placeholder.empty()

# --- ДВЕ ВКЛАДКИ ---
tab_search, tab_playlist = st.tabs(["🔍 Поиск музыки", f"❤️ Мой плейлист ({len(st.session_state.liked_ids)})"])

# ==========================================
# ВКЛАДКА 1: ПОИСК
# ==========================================
with tab_search:
    with st.form("profile"):
        c1, c2, c3 = st.columns(3)
        with c1:
            task = st.selectbox("Задача", ["учёба", "работа", "спорт", "чтение", "расслабление"])
            energy = st.slider("Энергия (1–5)", 1, 5, 3)
        with c2:
            vocal = st.radio("Вокал", ["no", "yes"], format_func=lambda x: "без вокала" if x == "no" else "можно")
            tempo = st.select_slider("Темп", ["медленный", "средний", "быстрый"], value="средний")
        with c3:
            mood = st.selectbox("Настроение", ["спокойное", "нейтральное", "бодрое", "грустное"])
            genre = st.selectbox("Жанр", ["любой", "lo-fi", "ambient", "classical", "electronic", "rap"])
        submitted = st.form_submit_button("Подобрать музыку")

    if submitted:
        query = f"Задача: {task}. Энергия: {energy}/5. Вокал: {vocal}. Темп: {tempo}. Настроение: {mood}. Жанр: {genre}."
        q_emb = model.encode(query, normalize_embeddings=True)

        # Персонализация
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

        unique_results = []
        seen_ids = set()
        for score, row in results:
            if row['track_id'] not in seen_ids:
                unique_results.append((score, row))
                seen_ids.add(row['track_id'])
            if len(unique_results) == 5:
                break

        st.session_state.results = unique_results

    if st.session_state.results:
        st.subheader("Ваша подборка")
        for score, row in st.session_state.results[:5]:
            with st.container(border=True):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**{row['title']}** — {row['artist']}")
                    st.caption(
                        f"Match: {score:.2f} · {row['genre']} · {row['bpm']:.0f} BPM · энергия {row['energy_1_5']}/5")

                    reasons = []
                    if vocal == "no" and row['vocal'] == "no": reasons.append("без вокала")
                    if abs(row['energy_1_5'] - energy) <= 1: reasons.append("энергия близка к вашей")
                    st.write(" • ".join(reasons) if reasons else "семантически близкое описание")

                    vals = [row["energy_1_5"] / 5, min(row["bpm"] / 160, 1), min(row["centroid"] / 4000, 1)]
                    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=["Энергия", "Темп", "Яркость", "Энергия"],
                                                    fill="toself", line_color="#FFFFFF"))  # <-- Белая линия
                    fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), showlegend=False,
                                      height=200, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_search_{row['track_id']}")

                with col2:
                    audio_path = DATA / row["file_path"]
                    if audio_path.exists():
                        st.audio(str(audio_path))
                        with open(audio_path, "rb") as file:
                            st.download_button(
                                label="📥 Скачать трек",
                                data=file,
                                file_name=f"{row['artist']} - {row['title']}.wav",
                                mime="audio/wav",
                                use_container_width=True,
                                key=f"dl_search_{row['track_id']}"
                            )
                    else:
                        st.warning("Аудиофайл не найден")

                    st.caption(f"Лицензия: {row['license']}")

                    btn_col1, btn_col2 = st.columns(2)
                    is_liked = str(row['track_id']) in st.session_state.liked_ids
                    is_disliked = str(row['track_id']) in st.session_state.disliked_ids

                    if btn_col1.button("❤️ В плейлисте" if is_liked else "🤍 В плейлист", key=f"like_{row['track_id']}",
                                       use_container_width=True):
                        if is_liked:
                            st.session_state.liked_ids.remove(str(row['track_id']))
                        else:
                            st.session_state.liked_ids.add(str(row['track_id']))
                            st.session_state.disliked_ids.discard(str(row['track_id']))
                            st.toast(f"Трек {row['title']} добавлен в плейлист!")
                        save_user_data()
                        st.rerun()

                    if btn_col2.button("❌ Не предлагать" if is_disliked else "👎 Не в тему",
                                       key=f"dislike_{row['track_id']}", use_container_width=True):
                        if is_disliked:
                            st.session_state.disliked_ids.remove(str(row['track_id']))
                        else:
                            st.session_state.disliked_ids.add(str(row['track_id']))
                            st.session_state.liked_ids.discard(str(row['track_id']))
                            st.toast(f"Мы учтем, что вам не понравился {row['title']}.")
                        save_user_data()
                        st.rerun()

# ==========================================
# ВКЛАДКА 2: ПЛЕЙЛИСТ
# ==========================================
with tab_playlist:
    st.subheader("❤️ Ваш личный плейлист")

    if not st.session_state.liked_ids:
        st.info(
            "Здесь пока пусто. Перейдите во вкладку «Поиск музыки» и нажмите «🤍 В плейлист» на треках, которые вам понравились.")
    else:
        liked_df = df[df['track_id'].astype(str).isin(st.session_state.liked_ids)]

        # Дашборд
        st.markdown("### 📊 Анализ вашего вкуса")
        c1, c2, c3 = st.columns(3)
        c1.metric("Треков в плейлисте", len(liked_df))
        c2.metric("Средний BPM", f"{liked_df['bpm'].mean():.0f}")
        c3.metric("Средняя энергия", f"{liked_df['energy_1_5'].mean():.1f} / 5")
        st.divider()

        for _, row in liked_df.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**{row['title']}** — {row['artist']}")
                    st.caption(f"{row['genre']} · {row['bpm']:.0f} BPM · энергия {row['energy_1_5']}/5")
                with col2:
                    audio_path = DATA / row["file_path"]
                    if audio_path.exists():
                        st.audio(str(audio_path))
                        with open(audio_path, "rb") as file:
                            st.download_button(
                                label="📥 Скачать трек",
                                data=file,
                                file_name=f"{row['artist']} - {row['title']}.wav",
                                mime="audio/wav",
                                use_container_width=True,
                                key=f"dl_pl_{row['track_id']}"
                            )
                    else:
                        st.warning("Файл не найден")

                    if st.button("🗑️ Удалить из плейлиста", key=f"remove_{row['track_id']}", use_container_width=True):
                        st.session_state.liked_ids.remove(str(row['track_id']))
                        save_user_data()
                        st.rerun()

        # Экспорт плейлиста
        st.markdown("---")
        csv_data = liked_df[['artist', 'title', 'genre', 'bpm']].to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📄 Скачать плейлист (CSV)",
            data=csv_data,
            file_name="my_studysound_playlist.csv",
            mime="text/csv",
            use_container_width=True
        )