package com.jasouza.stesis

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import com.google.android.gms.location.*
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.PrintWriter
import java.net.ServerSocket
import java.net.Socket
import java.net.URLDecoder
import java.util.concurrent.Executors

class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var sensorManager: SensorManager
    private var rotationSensor: Sensor? = null

    // Shared State Variables
    @Volatile private var currentLat: Double = 0.0
    @Volatile private var currentLong: Double = 0.0
    @Volatile private var currentAzimuth: Float = 0.0f

    // Compose State for Terminal UI
    private val terminalLogs = mutableStateListOf(
        "Stesis Terminal"
    )

    private val PORT = 8080
    private var isServerRunning = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Set up the Jetpack Compose UI
        setContent {
            TerminalScreen(logs = terminalLogs)
        }

        // Initialize Services
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        if (rotationSensor == null) {
            logToTerminal("WARNING: No rotation sensor found for compass.")
        }

        if (checkPermissions()) {
            startSensorsAndServer()
        } else {
            requestPermissions()
        }
    }

    // --- UI Logic ---
    @Composable
    fun TerminalScreen(logs: List<String>) {
        val listState = rememberLazyListState()

        // Auto-scroll to the bottom when new logs are added
        LaunchedEffect(logs.size) {
            if (logs.isNotEmpty()) {
                listState.animateScrollToItem(logs.size - 1)
            }
        }

        // The Black Terminal Background (75% opacity green text)
        LazyColumn(
            state = listState,
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
                .padding(16.dp)
        ) {
            items(logs) { log ->
                Text(
                    text = log,
                    color = Color.Green.copy(alpha = 0.75f), // 75% Opacity
                    fontFamily = FontFamily.Monospace,
                    fontSize = 14.sp,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
            }
        }
    }

    private fun logToTerminal(message: String) {
        runOnUiThread {
            terminalLogs.add(message)
        }
    }

    // --- Lifecycle & Sensors ---
    private fun startSensorsAndServer() {
        startLocationUpdates()
        rotationSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
        startWebServer()
        logToTerminal("> System initialized. Server listening on Port $PORT.")
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        isServerRunning = false
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_ROTATION_VECTOR) {
            val rotationMatrix = FloatArray(9)
            val orientationValues = FloatArray(3)

            SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
            SensorManager.getOrientation(rotationMatrix, orientationValues)

            var azimuth = Math.toDegrees(orientationValues[0].toDouble()).toFloat()
            if (azimuth < 0) azimuth += 360f
            currentAzimuth = azimuth
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // --- Location Logic ---
    @SuppressLint("MissingPermission")
    private fun startLocationUpdates() {
        val locationRequest = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 2000)
            .setWaitForAccurateLocation(false)
            .setMinUpdateIntervalMillis(1000)
            .build()

        fusedLocationClient.requestLocationUpdates(
            locationRequest,
            object : LocationCallback() {
                override fun onLocationResult(locationResult: LocationResult) {
                    for (location in locationResult.locations) {
                        currentLat = location.latitude
                        currentLong = location.longitude
                    }
                }
            },
            Looper.getMainLooper()
        )
    }

    // --- Web Server Logic ---
    private fun startWebServer() {
        if (isServerRunning) return
        isServerRunning = true

        Executors.newSingleThreadExecutor().execute {
            try {
                val serverSocket = ServerSocket(PORT)
                while (isServerRunning) {
                    val client = serverSocket.accept()
                    handleClient(client)
                }
            } catch (e: Exception) {
                logToTerminal("Server Error: ${e.message}")
            }
        }
    }

    private fun handleClient(socket: Socket) {
        try {
            val reader = BufferedReader(InputStreamReader(socket.getInputStream()))
            val output = PrintWriter(socket.getOutputStream())

            val requestLine = reader.readLine() ?: return
            val method = requestLine.split(" ")[0]
            val fullPath = requestLine.split(" ")[1]

            val path = fullPath.substringBefore("?")
            var jsonResponse = ""
            var statusCode = "200 OK"

            when (path) {
                "/stesis" -> {
                    logToTerminal("> STESIS CHECKED")
                    jsonResponse = """{"is_stesis": true}"""
                }
                "/gps" -> {
                    logToTerminal("> GET /gps requested")
                    jsonResponse = """{"latitude": $currentLat, "longitude": $currentLong}"""
                }
                "/compass" -> {
                    logToTerminal("> GET /compass requested")
                    jsonResponse = """{"heading_degrees": $currentAzimuth}"""
                }
                "/log" -> {
                    var logData = "Empty Log"

                    if (method == "POST") {
                        var contentLength = 0
                        var headerLine = reader.readLine()
                        while (!headerLine.isNullOrEmpty()) {
                            if (headerLine.lowercase().startsWith("content-length:")) {
                                contentLength = headerLine.split(":")[1].trim().toInt()
                            }
                            headerLine = reader.readLine()
                        }

                        if (contentLength > 0) {
                            val bodyChars = CharArray(contentLength)
                            reader.read(bodyChars, 0, contentLength)
                            logData = String(bodyChars)
                        }
                    } else if (method == "GET") {
                        val query = fullPath.substringAfter("?", "")
                        if (query.startsWith("data=")) {
                            logData = URLDecoder.decode(query.removePrefix("data="), "UTF-8")
                        }
                    }

                    logToTerminal(logData)
                    jsonResponse = """{"status": "Log received and printed"}"""
                }
                else -> {
                    statusCode = "404 Not Found"
                    jsonResponse = """{"error": "Endpoint not found"}"""
                }
            }

            output.println("HTTP/1.1 $statusCode")
            output.println("Content-Type: application/json")
            output.println("Access-Control-Allow-Origin: *")
            output.println("Content-Length: ${jsonResponse.length}")
            output.println("Connection: close")
            output.println()
            output.println(jsonResponse)

            output.flush()
            socket.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // --- Boilerplate Permissions Code ---
    private fun checkPermissions(): Boolean {
        return ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestPermissions() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
            101
        )
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 101 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startSensorsAndServer()
        } else {
            logToTerminal("ERROR: Location permission denied. GPS will not work.")
        }
    }
}