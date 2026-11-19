<?php
declare(strict_types=1);

// Optional absolute path to chat.php. Store it outside all public document roots.
const OSERAI_PRIVATE_CONFIG = '';
const UNAVAILABLE = 'Chat is coming online. Please check back soon.';

ini_set('display_errors', '0');
header('Cache-Control: no-store, no-transform');
header('X-Content-Type-Options: nosniff');

function json_response(array $body, int $status = 200): never {
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode($body, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
    exit;
}

function private_config_path(): string {
    $override = OSERAI_PRIVATE_CONFIG ?: getenv('OSERAI_CHAT_CONFIG');
    if ($override) { return $override; }

    $suffix = '/.config/oserai/chat.php';
    $home = rtrim((string)getenv('HOME'), '/');
    if ($home !== '' && is_file($home . $suffix) && is_readable($home . $suffix)) {
        return $home . $suffix;
    }

    // cPanel's PHP handler can omit HOME. Its document roots live under the account home.
    if (preg_match('~\A(/home[0-9]*/[^/]+)/~', __DIR__, $matches) === 1) {
        return $matches[1] . $suffix;
    }
    return '';
}

function quota_payload(mixed $value): ?array {
    if (!is_array($value) || !is_int($value['limit'] ?? null) || !is_int($value['remaining'] ?? null)
        || !is_string($value['resets_at'] ?? null)) {
        return null;
    }
    return ['limit' => $value['limit'], 'remaining' => $value['remaining'], 'resets_at' => $value['resets_at']];
}

function change_account(string $csrf, string $credential): never {
    session_start();
    if (!hash_equals($_SESSION['csrf'] ?? '', $csrf)) {
        session_write_close();
        json_response(['message' => 'Please refresh the page and try again.'], 403);
    }
    if (!session_regenerate_id(true)) {
        session_write_close();
        json_response(['message' => UNAVAILABLE], 503);
    }
    if ($credential === '') {
        unset($_SESSION['account']);
    } else {
        $_SESSION['account'] = $credential;
    }
    $_SESSION['csrf'] = bin2hex(random_bytes(32));
    $newCsrf = $_SESSION['csrf'];
    session_write_close();
    json_response(['csrf_token' => $newCsrf]);
}

$configPath = private_config_path();
$config = is_file($configPath) && is_readable($configPath) ? require $configPath : [];
if (!is_array($config)) { $config = []; }
$origin = $config['allowed_origin'] ?? 'https://chat.oserai.network';
$secure = str_starts_with($origin, 'https://');
session_name('oserai_session');
ini_set('session.use_strict_mode', '1');
session_set_cookie_params(['lifetime' => 0, 'path' => '/', 'secure' => $secure, 'httponly' => true, 'samesite' => 'Lax']);
session_start();
$_SESSION['csrf'] ??= bin2hex(random_bytes(32));
$csrf = $_SESSION['csrf'];
$account = (string)($_SESSION['account'] ?? '');
$visitor = hash('sha256', session_id());
// A long stream must not lock out status requests or another browser tab.
session_write_close();

$action = $_GET['action'] ?? 'status';
$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
if (!(($action === 'status' && $method === 'GET') || ($method === 'POST' && in_array($action, ['chat', 'login', 'logout'], true)))) {
    json_response(['message' => 'Request not supported.'], 405);
}
if ($method === 'POST') {
    $requestOrigin = $_SERVER['HTTP_ORIGIN'] ?? '';
    if (($requestOrigin !== '' && $requestOrigin !== $origin)
        || ($_SERVER['HTTP_SEC_FETCH_SITE'] ?? '') === 'cross-site'
        || !hash_equals($csrf, $_SERVER['HTTP_X_CSRF_TOKEN'] ?? '')) {
        json_response(['message' => 'Please refresh the page and try again.'], 403);
    }
    if (!str_starts_with(strtolower($_SERVER['CONTENT_TYPE'] ?? ''), 'application/json')) {
        json_response(['message' => 'Please send a chat message.'], 415);
    }
}
if ($action === 'logout') {
    change_account($csrf, '');
}

$gateway = rtrim((string)($config['gateway_url'] ?? ''), '/');
$key = (string)($config['gateway_key'] ?? '');
$url = parse_url($gateway);
$local = ($config['allow_local_http'] ?? false) && in_array($url['host'] ?? '', ['127.0.0.1', 'localhost', '::1'], true);
$hasCurl = function_exists('curl_init');
$configured = is_array($url) && isset($url['host'])
    && (($url['scheme'] ?? '') === 'https' || ($local && ($url['scheme'] ?? '') === 'http'))
    && !isset($url['user']) && !isset($url['pass']) && !isset($url['query']) && !isset($url['fragment'])
    && strlen($key) >= 32 && !str_contains($key, "\n") && !str_contains($key, "\r")
    && $hasCurl;
if (!$configured) {
    if (!$hasCurl) {
        error_log('OserAI Chat: the PHP cURL extension is unavailable.');
    } elseif (!$config) {
        error_log('OserAI Chat: the private chat.php settings could not be loaded.');
    } else {
        error_log('OserAI Chat: check the gateway URL and gateway key in the private chat.php settings.');
    }
    $body = ['ready' => false, 'model_label' => 'OserAI Chat', 'message' => UNAVAILABLE];
    if ($action === 'status') { $body['csrf_token'] = $csrf; }
    json_response($body, $action === 'status' ? 200 : 503);
}

$network = hash_hmac('sha256', $_SERVER['REMOTE_ADDR'] ?? 'unknown', $key);
$headers = ['Authorization: Bearer ' . $key, 'X-Visitor-ID: ' . $visitor, 'X-Visitor-Network: ' . $network];
$credentialPattern = '~\A[a-z0-9][a-z0-9_-]{0,39}\.[a-f0-9]{64}\z~';
if (preg_match($credentialPattern, $account) === 1) {
    $headers[] = 'X-Chat-Account: ' . $account;
}
$curl = curl_init($gateway . '/api/' . $action);
curl_setopt_array($curl, [
    CURLOPT_HTTPHEADER => array_merge($headers, ['Accept: application/json']),
    CURLOPT_CONNECTTIMEOUT => 10,
    CURLOPT_TIMEOUT => $action === 'chat' ? 175 : 15,
    CURLOPT_FOLLOWLOCATION => false,
    CURLOPT_SSL_VERIFYPEER => true,
    CURLOPT_SSL_VERIFYHOST => 2,
    CURLOPT_PROTOCOLS => CURLPROTO_HTTPS | ($local ? CURLPROTO_HTTP : 0),
]);

if ($action === 'status') {
    curl_setopt($curl, CURLOPT_RETURNTRANSFER, true);
    $raw = curl_exec($curl);
    $status = (int)curl_getinfo($curl, CURLINFO_RESPONSE_CODE);
    curl_close($curl);
    $payload = is_string($raw) ? json_decode($raw, true) : null;
    $valid = $status === 200 && is_array($payload);
    $ready = $valid && ($payload['ready'] ?? false) === true;
    $signedIn = $valid && ($payload['signed_in'] ?? false) === true;
    json_response([
        'ready' => $ready,
        'model_label' => $ready ? substr((string)($payload['model_label'] ?? 'OserAI Chat'), 0, 80) : 'OserAI Chat',
        'max_input_chars' => $ready ? min(16000, max(1, (int)($payload['max_input_chars'] ?? 16000))) : 16000,
        'message' => $ready ? 'Ready to chat' : UNAVAILABLE,
        'csrf_token' => $csrf,
        'accounts_required' => $valid ? ($payload['accounts_required'] ?? false) === true : null,
        'signed_in' => $valid ? $signedIn : null,
        'quota' => $signedIn ? quota_payload($payload['quota'] ?? null) : null,
    ]);
}

if ($action === 'login') {
    $raw = file_get_contents('php://input', false, null, 0, 1025);
    $body = is_string($raw) && strlen($raw) <= 1024 ? json_decode($raw, true) : null;
    $code = is_array($body) && is_string($body['code'] ?? null) ? trim($body['code']) : '';
    if (preg_match('~\A[A-Za-z0-9_-]{43}\z~', $code) !== 1) {
        json_response(['message' => 'Enter the private access code you received.'], 422);
    }
    curl_setopt_array($curl, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => json_encode(['code' => $code]),
        CURLOPT_HTTPHEADER => array_merge($headers, ['Accept: application/json', 'Content-Type: application/json']),
    ]);
    $raw = curl_exec($curl);
    $status = (int)curl_getinfo($curl, CURLINFO_RESPONSE_CODE);
    curl_close($curl);
    $payload = is_string($raw) ? json_decode($raw, true) : null;
    $credential = is_array($payload) && is_string($payload['account_credential'] ?? null)
        ? $payload['account_credential'] : '';
    if ($status === 200 && preg_match($credentialPattern, $credential) === 1) {
        change_account($csrf, $credential);
    }
    if ($status === 401) {
        json_response(['message' => 'That access code is invalid.'], 401);
    }
    json_response(['message' => UNAVAILABLE], 503);
}

if ((int)($_SERVER['CONTENT_LENGTH'] ?? 0) > 160000) {
    json_response(['message' => 'This conversation is full. Please start a new chat.'], 413);
}
$raw = file_get_contents('php://input', false, null, 0, 160001);
if ($raw === false || strlen($raw) > 160000) {
    json_response(['message' => 'This conversation is full. Please start a new chat.'], 413);
}
$body = json_decode($raw, true);
if (!is_array($body) || !isset($body['messages']) || !is_array($body['messages']) || count($body['messages']) > 64) {
    json_response(['message' => 'Please send a chat message.'], 422);
}

@set_time_limit(185);
ini_set('zlib.output_compression', '0');
while (ob_get_level() > 0) { ob_end_clean(); }
ignore_user_abort(false);
$streamStarted = false;
$streamGateway = ($config['gateway_stream'] ?? true) === true;
$upstreamStatus = 0;
$contentType = '';
$errorBody = '';
curl_setopt_array($curl, [
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => json_encode(['messages' => $body['messages'], 'stream' => $streamGateway], JSON_UNESCAPED_UNICODE),
    CURLOPT_HTTPHEADER => array_merge($headers, ['Content-Type: application/json', 'Accept: ' . ($streamGateway ? 'text/event-stream' : 'application/json')]),
    CURLOPT_HEADERFUNCTION => function ($handle, string $line) use (&$upstreamStatus, &$contentType): int {
        if (preg_match('/^HTTP\/\S+\s+(\d+)/', $line, $matches)) { $upstreamStatus = (int)$matches[1]; }
        if (stripos($line, 'content-type:') === 0) { $contentType = strtolower(trim(substr($line, 13))); }
        return strlen($line);
    },
    CURLOPT_WRITEFUNCTION => function ($handle, string $chunk) use (&$streamStarted, &$upstreamStatus, &$contentType, &$errorBody, $streamGateway): int {
        if (!$streamGateway) {
            if (strlen($errorBody) + strlen($chunk) > 500000) { return 0; }
            $errorBody .= $chunk;
            return connection_aborted() ? 0 : strlen($chunk);
        }
        if ($upstreamStatus !== 200 || !str_starts_with($contentType, 'text/event-stream')) {
            if (strlen($errorBody) < 65536) { $errorBody .= $chunk; }
            return strlen($chunk);
        }
        if (connection_aborted()) { return 0; }
        if (!$streamStarted) {
            header('Content-Type: text/event-stream; charset=utf-8');
            header('X-Accel-Buffering: no');
            $streamStarted = true;
        }
        echo $chunk;
        flush();
        return connection_aborted() ? 0 : strlen($chunk);
    },
]);
$result = curl_exec($curl);
curl_close($curl);
if (connection_aborted()) { exit; }
if (!$streamGateway && $result !== false && $upstreamStatus === 200 && str_starts_with($contentType, 'application/json')) {
    $payload = json_decode($errorBody, true);
    if (is_array($payload) && is_string($payload['text'] ?? null) && trim($payload['text']) !== ''
        && strlen($payload['text']) <= 400000 && is_numeric($payload['elapsed_s'] ?? null)) {
        header('Content-Type: text/event-stream; charset=utf-8');
        header('X-Accel-Buffering: no');
        $count = $payload['completion_tokens'] ?? null;
        foreach ([
            'start' => ['request_id' => (string)($payload['request_id'] ?? ''), 'model_label' => 'OserAI Chat'],
            'token' => ['text' => $payload['text']],
            'done' => ['completion_tokens' => is_int($count) && $count >= 0 ? $count : null, 'elapsed_s' => (float)$payload['elapsed_s'],
                       'quota' => quota_payload($payload['quota'] ?? null)],
        ] as $name => $data) {
            echo 'event: ' . $name . "\ndata: " . json_encode($data, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) . "\n\n";
        }
        flush();
        exit;
    }
}
if (!$streamStarted) {
    $payload = json_decode($errorBody, true);
    $message = is_array($payload) && is_string($payload['message'] ?? null) ? $payload['message'] : UNAVAILABLE;
    json_response(['message' => $message], in_array($upstreamStatus, [400, 401, 403, 413, 422, 429, 503], true) ? $upstreamStatus : 503);
}
if ($result === false) {
    echo "event: error\ndata: " . json_encode(['message' => 'The reply was interrupted. Please try again in a moment.']) . "\n\n";
    flush();
}
