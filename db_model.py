# db_model.py
import sqlite3
from datetime import datetime, timedelta
import random

class OrderDatabase:
    def __init__(self, db_name="online_sales.db"):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def create_table(self):
        """Create orders table"""
        self.connect()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cust_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                order_number TEXT UNIQUE NOT NULL,
                amount REAL NOT NULL,
                payment_status TEXT NOT NULL,
                order_status TEXT NOT NULL,
                delivery_date TEXT,
                expected_delivery_date TEXT NOT NULL,
                delay_reason TEXT
            )
        ''')
        self.conn.commit()
        self.close()
        print("✅ Table 'orders' created successfully")
    
    def insert_sample_data(self):
        """Insert 10 sample orders"""
        self.connect()
        
        customers = [
            ("John Smith", "555-0101"),
            ("Sarah Johnson", "555-0102"),
            ("Mike Davis", "555-0103"),
            ("Emily Brown", "555-0104"),
            ("David Wilson", "555-0105"),
            ("Lisa Anderson", "555-0106"),
            ("Tom Martinez", "555-0107"),
            ("Anna Taylor", "555-0108"),
            ("Chris Lee", "555-0109"),
            ("Maria Garcia", "555-0110")
        ]
        
        payment_statuses = ["Paid", "Pending", "Failed", "Refunded"]
        order_statuses = ["Processing", "Shipped", "Delivered", "Delayed", "Cancelled"]
        delay_reasons = [
            None,
            "Weather conditions",
            "Out of stock",
            "Address issue",
            "Courier delay",
            "Customs clearance"
        ]
        
        base_date = datetime.now()
        
        sample_orders = []
        for i, (name, phone) in enumerate(customers, 1):
            order_num = f"ORD{base_date.year}{str(i).zfill(6)}"
            amount = round(random.uniform(50, 500), 2)
            payment = random.choice(payment_statuses)
            status = random.choice(order_statuses)
            
            # Dates
            expected_delivery = (base_date + timedelta(days=random.randint(3, 7))).strftime("%Y-%m-%d")
            
            if status == "Delivered":
                delivery = (base_date - timedelta(days=random.randint(1, 3))).strftime("%Y-%m-%d")
                delay = None
            elif status == "Delayed":
                delivery = None
                delay = random.choice([r for r in delay_reasons if r])
            else:
                delivery = None
                delay = None
            
            sample_orders.append((
                name, phone, order_num, amount, payment, status,
                delivery, expected_delivery, delay
            ))
        
        self.cursor.executemany('''
            INSERT INTO orders (cust_name, phone, order_number, amount, payment_status, 
                              order_status, delivery_date, expected_delivery_date, delay_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', sample_orders)
        
        self.conn.commit()
        self.close()
        print(f"✅ Inserted {len(sample_orders)} sample orders")
    
    def get_order_by_number(self, order_number):
        """Get order details by order number"""
        self.connect()
        self.cursor.execute('''
            SELECT * FROM orders WHERE order_number = ?
        ''', (order_number,))
        result = self.cursor.fetchone()
        self.close()
        return result
    
    def get_order_by_phone(self, phone):
        """Get orders by phone number"""
        self.connect()
        self.cursor.execute('''
            SELECT * FROM orders WHERE phone = ?
        ''', (phone,))
        results = self.cursor.fetchall()
        self.close()
        return results
    
    def get_orders_by_customer_name(self, customer_name):
        """Get orders by exact customer name"""
        self.connect()
        self.cursor.execute('''
            SELECT * FROM orders WHERE cust_name = ?
        ''', (customer_name,))
        results = self.cursor.fetchall()
        self.close()
        return results
    
    def get_orders_by_customer_id(self, customer_id):
        """Get orders by exact customer name"""
        self.connect()
        self.cursor.execute('''
            SELECT * FROM orders WHERE cust_id = ?
        ''', (customer_id,))
        results = self.cursor.fetchall()
        self.close()
        return results
    
    def get_all_orders(self):
        """Get all orders"""
        self.connect()
        self.cursor.execute('SELECT * FROM orders')
        results = self.cursor.fetchall()
        self.close()
        return results
    
    def search_by_customer_name(self, name):
        """Search orders by customer name (partial match)"""
        self.connect()
        self.cursor.execute('''
            SELECT * FROM orders WHERE cust_name LIKE ?
        ''', (f'%{name}%',))
        results = self.cursor.fetchall()
        self.close()
        return results


# Initialize database
if __name__ == "__main__":
    db = OrderDatabase()
    
    # Create table
    db.create_table()
    
    # Insert sample data
    db.insert_sample_data()
    
    # Show sample data
    print("\n📦 Sample Orders:")
    orders = db.get_all_orders()
    for order in orders[:3]:
        print(f"  Order: {order[3]} | Customer: {order[1]} | Status: {order[6]} | Amount: ${order[4]}")
    
    print(f"\n✅ Database setup complete! Total orders: {len(orders)}")