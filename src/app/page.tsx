import { connectToDatabase } from "./backend/mongo"; // Adjust the path as needed

export default function Home() {
  async function handleMongoConnection() {
    try {
      const db = await connectToDatabase();
      console.log("Database instance:", db);
      // Perform additional actions with the database instance if needed
    } catch (error) {
      console.error("Error connecting to MongoDB:", error);
    }
  }

  handleMongoConnection(); // Call the function here or on an event like a button click

  return (
    <div>
      <h1>Welcome to StockSim</h1>
      <p>Check the console for MongoDB connection status.</p>
    </div>
  );
}
