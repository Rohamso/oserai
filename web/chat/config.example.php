<?php
// Save the real copy OUTSIDE the website document root:
// /home/YOUR_CPANEL_USERNAME/.config/oserai/chat.php
return [
    'gateway_url' => 'https://YOUR-CHAT-GATEWAY-HOST',
    'gateway_key' => 'THE-SAME-RANDOM-KEY-AS-OSERAI_CHAT_GATEWAY_KEY',
    'allowed_origin' => 'https://chat.oserai.network',
    'allow_local_http' => false,
    // Set false for a testing tunnel that does not support streaming/SSE.
    'gateway_stream' => true,
];
