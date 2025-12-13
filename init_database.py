import sqlite3

def init_student_database():
    """Initialize the STUDENTS table with auto-increment ID"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Create STUDENTS table with auto-increment ID
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS STUDENTS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL,
            age INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ STUDENTS table created successfully with auto-increment ID")

if __name__ == "__main__":
    init_student_database()
    print("📋 Database initialization complete!")