import { connectToDatabase } from "../backend/mongo"; // Adjust the path as needed

import React from "react";
import Search from "../components/Search"
import HomePage from "../components/MainUserPage";
import Login from "../components/Login";
//import Footer from "../components/Footer";

export default function Home() {
  return (
      <Login />
  );
}
