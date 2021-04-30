<template>
  <div>

<div>
<b-link href="http://localhost:7776/" target="_blank">piSTAR Launcher</b-link>
</div>
<div class="mt-2"></div>
<div>
<b-link href="http://localhost:8265/" target="_blank">Ray Dashboard</b-link>
<div class="mt-2"></div>
<div>
<b-link href="http://localhost:7777/api/reload_default_data" target="_blank">reload_default_data</b-link>
</div>
</div>
    <b-container fluid>
      <b-row class="pt-4">
        <b-col>
          {{ message }}
        </b-col>
      </b-row>

      <!-- <b-row class="pt-4">
        <b-col>
          <b-card>
            <b-card-text>
              <b-button
                @click="adminCommand('reset_backend')"
                variant="warning"
              >
                RESTART RAY</b-button
              >
            </b-card-text>
          </b-card>
        </b-col>
      </b-row>
      <b-row class="pt-4">
        <b-col>
          <b-card>
            <b-card-text>
              <b-button
                @click="adminCommand('reset_core_data')"
                variant="danger"
              >
                RESET CORE DATA</b-button
              >
              Warning: all pistar data will be deleted.
            </b-card-text>
          </b-card>
        </b-col>
      </b-row> -->

      <b-row class="pt-4">
        <b-col>
          <div class="pt-1" v-for="(item, name) in data" v-bind:key="name">
            <div>{{ name }}</div>
            <div>{{ item }}</div>
          </div>
        </b-col>
      </b-row>
    </b-container>
  </div>
</template>

<script>
// @ is an alias to /src
import axios from "axios"; import {appConfig} from "../app.config";

export default {
  name: "Preferences",
  components: {
    // SessionList
  },
  data() {
    return {
      searchQuery: "",
      items: [],
      error: "",
      message: "",
      data: {},
    };
  },
  methods: {
    adminCommand(cmd) {
      this.submitting = true;
      axios
        .get(`${appConfig.API_URL}/api/admin_command/` + cmd)
        .then((response) => {
          this.message = response.data["message"];

          console.log(response);
          this.submitting = false;
        })
        .catch((error) => {
          this.message = error;
          this.submitting = false;
        });
    },

    loadAdminData() {
      axios
        .get(`${appConfig.API_URL}/api/admin_data/`)
        .then((response) => {
          this.data = response.data["data"];
        })
        .catch((error) => {
          this.message = error;
        });
    },
  },
  computed: {},
  // Fetches posts when the component is created.
  created() {
    //
    this.loadAdminData();
  },
};
</script>