const express = require("express");
const bodyParser = require("body-parser");
const { spawn } = require("child_process"); // To run the Python script
const fs = require("fs"); // To read the JSON file
const cors = require("cors");

const multer = require("multer");
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "./uploads");
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname);
  },
});

const upload = multer({ storage: storage });

const app = express();

// Increase the limit for JSON and URL-encoded data
app.use(bodyParser.json({limit: '50mb'}));
app.use(bodyParser.urlencoded({limit: '50mb', extended: true}));

require("dotenv").config();
const mysql = require('mysql2/promise');

// MySQL Connection Pool
const pool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_DATABASE,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

app.use(
  cors({
    origin: "http://localhost:5173",
    credentials: true,
  })
);

app.post("/api/transform-data", upload.single("file"), (req, res) => {
  try {
    const file = req.file;
    const selectedOption = req.body.selectedOption;
    const filterSize = req.body.filterSize;
    const windowType = req.body.windowType;
    const overlap = req.body.overlap;

    // console.log("File uploaded:", file);
    // console.log("Selected Option:", selectedOption);
    // console.log("Filter Size:", filterSize);
    // console.log("Window Type:", windowType);
    // console.log("Overlap:", overlap);

    // Save the uploaded file to a temporary location
    const tempFilePath = req.file.path;
    fs.readFile(tempFilePath, (err) => {
      if (err) {
        console.error(err);
      } else {
        // Run Python script using `spawn`
        const pythonProcess = spawn("python", [
          "./script.py",
          "transform",
          tempFilePath,
          "./output.json",
          selectedOption,
          filterSize,
          windowType,
          overlap,
        ]);

        // Capture any output from the Python script (optional for logging)
        pythonProcess.stdout.on("data", (data) => {
          console.log(`Python Output: ${data}`);
        });

        // Capture errors from the Python script
        pythonProcess.stderr.on("data", (data) => {
          console.error(`Python Error: ${data}`);
        });

        // When the Python script finishes
        pythonProcess.on("close", (code) => {
          console.log(`Python script exited with code ${code}`);

          // Read the generated JSON file
          fs.readFile("./output.json", "utf8", (err, jsonData) => {
            if (err) {
              return res
                .status(500)
                .json({ error: "Error reading the output JSON file" });
            }

            // Send the JSON data as a response
            // try {
            //   const fftData = JSON.parse(jsonData);

            //   // Insert the FFT data into PostgreSQL
            //   const query = `
            //             INSERT INTO fft_results (analysis_data)
            //             VALUES ($1)
            //             RETURNING id;
            //         `;
            //   const values = [jsonData];

            //   pool.query(query, values, (error, result) => {
            //     if (error) {
            //       console.log("Error inserting data into PostgreSQL:", error);
            //       return res
            //         .status(500)
            //         .json({ error: "Error inserting data into PostgreSQL" });
            //     }
            //     // Respond with the inserted row's ID or any other success message
            //     res.json({ success: true, id: result.rows[0].id });
            //   });
            // } catch (parseError) {
            //   res
            //     .status(500)
            //     .json({ error: "Error parsing the output JSON file" });
            // }

            // Send the JSON data as a response and get it downloaded
            // Send the JSON data as a response and get it downloaded
            res.set(
              "Content-Disposition",
              `attachment; filename="${selectedOption}-transformed.json"`
            );
            res.set("Content-Type", "application/json");
            res.send(jsonData);
          });
        });
      }
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error processing the request" });
  }
});

app.get("/api/get-filtered-data", async (req, res) => {
  try {
    const query = `
      SELECT x_data, y_data, z_data, data_index, created_at
      FROM filtered_data
      ORDER BY data_index ASC;
    `;

    const [results] = await pool.query(query);
    
    if (results.length > 0) {
      let responseData = {
        xData: [],
        yData: [],
        zData: [],
        createdAt: results[0].created_at
      };

      results.forEach(row => {
        responseData.xData.push(row.x_data);
        responseData.yData.push(row.y_data);
        responseData.zData.push(row.z_data);
      });

      res.json(responseData);
    } else {
      res.status(404).json({ error: "No data found" });
    }
  } catch (error) {
    console.error("Error retrieving data from MySQL:", error);
    res.status(500).json({ error: "Error retrieving data from MySQL", details: error.message, stack: error.stack });
  }
});

app.post("/api/filter", async (req, res) => {
  try {
    // Read the output_filter.json file
    const jsonData = fs.readFileSync('./output_filter.json', 'utf8');
    const filterData = JSON.parse(jsonData);

    // Validate the input data
    if (!filterData.x_data || !filterData.y_data || !filterData.z_data) {
      res.status(400).send({ message: "Invalid input data" });
      return;
    }

    console.log('Received data:', filterData);

    const result = await storeFilteredData(filterData);
    console.log('Store result:', result);

    res.json({ message: "Data stored successfully", result });
  } catch (error) {
    console.error("Error in filter process:", error);
    res.status(500).json({ error: "Error in filter process", details: error.message });
  }
});

async function storeFilteredData(filteredData) {
  try {
    console.log('Filtered Data:', filteredData);

    // Ensure each data array exists
    const x_data = filteredData.x_data || [];
    const y_data = filteredData.y_data || [];
    const z_data = filteredData.z_data || [];

    // Determine the maximum length of the arrays
    const maxLength = Math.max(x_data.length, y_data.length, z_data.length);

    // Prepare the query for bulk insert
    const query = 'INSERT INTO filtered_data (x_data, y_data, z_data, data_index) VALUES ?';
    const values = [];

    for (let i = 0; i < maxLength; i++) {
      values.push([
        x_data[i] || null,
        y_data[i] || null,
        z_data[i] || null,
        i
      ]);
    }

    const result = await pool.query(query, [values]);
    console.log('Insert Result:', result);
    return result;
  } catch (error) {
    console.error("Error storing filtered data:", error);
    throw error;
  }
}

// Function to retrieve filtered data from MySQL
async function getFilteredData() {
  try {
    const [rows] = await pool.query('SELECT * FROM filtered_data ORDER BY id DESC LIMIT 1');
    if (rows.length > 0) {
      const row = rows[0];
      return {
        id: row.id,
        xData: JSON.parse(row.x_data),
        yData: JSON.parse(row.y_data),
        zData: JSON.parse(row.z_data),
        createdAt: row.created_at
      };
    }
    return null;
  } catch (error) {
    console.error("Error retrieving filtered data:", error);
    throw error;
  }
}

app.get("/run-fft", (req, res) => {
  // ... (previous code remains unchanged)

  pythonProcess.on("close", (code) => {
    console.log(`Python script exited with code ${code}`);

    fs.readFile(outputFilePath, "utf8", (err, jsonData) => {
      if (err) {
        return res.status(500).json({ error: "Error reading the output JSON file" });
      }

      try {
        const fftData = JSON.parse(jsonData);

        // Ensure that the data has the correct structure
        if (!fftData.data) {
          return res.status(400).json({ error: "Invalid data structure" });
        }

        // Insert the FFT data into MySQL
        const query = `
          INSERT INTO filtered_data (x_data, y_data, z_data) 
          VALUES (?, ?, ?);
        `;
        const values = [
          JSON.stringify(fftData.data.x || []),
          JSON.stringify(fftData.data.y || []),
          JSON.stringify(fftData.data.z || [])
        ];

        pool.query(query, values, (error, result) => {
          if (error) {
            console.log("Error inserting data into MySQL:", error);
            return res.status(500).json({ error: "Error inserting data into MySQL" });
          }
          // Respond with the inserted row's ID or any other success message
          res.json({ success: true, id: result.insertId });
        });
      } catch (parseError) {
        res.status(500).json({ error: "Error parsing the output JSON file" });
      }
    });
  });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

app.get('/some-route', async (req, res) => {
  try {
    const [rows] = await pool.query('YOUR SQL QUERY HERE', [param1, param2]);
    res.json(rows);
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'An error occurred' });
  }
});

app.get("/test-db-connection", async (req, res) => {
  try {
    const [rows] = await pool.query('SELECT 1');
    res.json({ message: "Database connection successful", result: rows });
  } catch (error) {
    console.error("Database connection error:", error);
    res.status(500).json({ error: "Database connection error", details: error.message, stack: error.stack });
  }
});
