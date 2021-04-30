
let API_URL = process.env.VUE_APP_BACKEND_URL
if (typeof API_URL === 'undefined') {
    API_URL = ""
}

const appConfig= {
  API_URL: API_URL,
}
function fetchSettings(){
    return fetch(API_URL + "/api/config").then(response => response.json() )
}

export {appConfig, fetchSettings};
