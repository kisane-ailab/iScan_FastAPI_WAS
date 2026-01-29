const { Client } = require("ssh2");
const fs = require("fs");
const path = require("path");

// Load environment variables from .env.deploy
const envFile = fs.readFileSync(".env.deploy", "utf-8");
const envVars = {};
envFile.split("\n").forEach((line) => {
  const trimmed = line.trim();
  if (trimmed && !trimmed.startsWith("#")) {
    const [key, ...valueParts] = trimmed.split("=");
    if (key) {
      envVars[key.trim()] = valueParts.join("=").trim();
    }
  }
});

const config = {
  host: envVars.DEPLOY_HOST,
  port: parseInt(envVars.DEPLOY_PORT || "22", 10),
  username: envVars.DEPLOY_USERNAME,
  password: envVars.DEPLOY_PASSWORD,
};

const remotePath = envVars.DEPLOY_REMOTE_PATH;
const appPort = envVars.APP_PORT;
const appUrl = envVars.APP_URL;

const localFile = "iscan-was-manager-standalone.tar.gz";
const remoteFile = `${remotePath}/${localFile}`;

console.log("Configuration:");
console.log(`  Host: ${config.host}`);
console.log(`  Remote Path: ${remotePath}`);
console.log(`  App Port: ${appPort}`);
console.log("");
console.log("Connecting to server...");

const conn = new Client();

conn.on("ready", () => {
  console.log("Connected!");

  // Create remote directory if it doesn't exist
  conn.exec(`mkdir -p ${remotePath}`, (err) => {
    if (err) {
      console.error("Failed to create remote directory:", err.message);
      conn.end();
      process.exit(1);
    }

    // SFTP upload
    conn.sftp((err, sftp) => {
      if (err) {
        console.error("SFTP error:", err.message);
        conn.end();
        process.exit(1);
      }

      console.log("Uploading...");
      const readStream = fs.createReadStream(localFile);
      const writeStream = sftp.createWriteStream(remoteFile);

      writeStream.on("close", () => {
        console.log("Upload successful!");

        // Deploy commands
        const deployCmd = [
          `cd ${remotePath}`,
          `tar -xzf ${localFile}`,
          `cp -r .next/static .next/standalone/.next/`,
          `cp -r public .next/standalone/`,
          `cd .next/standalone`,
          `pm2 delete iscan-was-manager-web 2>/dev/null || true`,
          `PORT=${appPort} HOSTNAME=0.0.0.0 pm2 start server.js --name 'iscan-was-manager-web'`,
          `pm2 save`,
        ].join(" && ");

        console.log("Deploying...");
        conn.exec(deployCmd, (err, stream) => {
          if (err) {
            console.error("Exec error:", err.message);
            conn.end();
            process.exit(1);
          }

          stream
            .on("close", (code) => {
              console.log("");
              console.log(`Done! ${appUrl}`);
              conn.end();
              process.exit(code || 0);
            })
            .on("data", (data) => {
              console.log(data.toString());
            })
            .stderr.on("data", (data) => {
              console.error(data.toString());
            });
        });
      });

      writeStream.on("error", (err) => {
        console.error("Upload error:", err.message);
        conn.end();
        process.exit(1);
      });

      readStream.pipe(writeStream);
    });
  });
});

conn.on("error", (err) => {
  console.error("Connection error:", err.message);
  process.exit(1);
});

conn.connect(config);
