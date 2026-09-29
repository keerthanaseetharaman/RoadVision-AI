import sqlite3

DATABASE_NAME = "roadvision.db"


def create_database():

    conn = sqlite3.connect(DATABASE_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            damage_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            affected_area REAL NOT NULL,
            latitude REAL,
            longitude REAL,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_detection(
    damage_type,
    confidence,
    severity,
    affected_area,
    latitude,
    longitude
):

    conn = sqlite3.connect(DATABASE_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO detections (
            damage_type,
            confidence,
            severity,
            affected_area,
            latitude,
            longitude
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        damage_type,
        confidence,
        severity,
        affected_area,
        latitude,
        longitude
    ))

    conn.commit()
    conn.close()


def get_detections():

    conn = sqlite3.connect(DATABASE_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            damage_type,
            confidence,
            severity,
            affected_area,
            latitude,
            longitude,
            detected_at
        FROM detections
        ORDER BY detected_at DESC
    """)

    records = cursor.fetchall()

    conn.close()

    return records