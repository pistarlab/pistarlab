<template>
<div v-if="item">
    <h2><i class="fa fa-cube"></i> Session: {{item.ident}}, Episode: {{selectedEpisodeId}}</h2>
    <b-container fluid>
        <b-row>
            <b-col>
                <b-alert v-if="message" show variant="warning">{{ message }}</b-alert>
            </b-col>
        </b-row>
        <b-row>
            <b-col>
                <b-navbar toggleable="lg" type="light">
                    <b-button-toolbar aria-label="Toolbar with button groups and input groups">
                        <b-button-group class="mr-1">
                            <b-button :to="`/session/view/${item.ident}`">
                                <i class="fas fa-arrow-up"></i> Session
                            </b-button>
                        </b-button-group>

                    </b-button-toolbar>
                </b-navbar>
                <b-row>
                    <b-col>
                        <b-form-group label="Episode">
                            <b-form-select :options="episodeList" @change="updateEpisode()" v-model="selectedEpisodeId">
                            </b-form-select>
                        </b-form-group>
                    </b-col>
                </b-row>
            </b-col>
        </b-row>
        <div class="mt-2"></div>
        <b-row>
            <b-col>
                <b-form-group label="FPS">
                 <b-form-select  @change="loadData()" v-model="fps" :options="fpsOptions" size="sm" class="mt-3 mb-3"></b-form-select>
                </b-form-group>
                <embed :src="videoURL" type="video/mp4" height="300px">
                
            </b-col>
        </b-row>

        <b-row>
            <b-col>
                <b-card>
                    <b-card-text>
                        <b-table striped hover table-busy :items="episode" :fields="fields" :dark="false" :small="false" :bordered="false" :outlined="false" :borderless="false" :no-provider-paging="true" :per-page="perPage" :current-page="currentPage">
                            <template v-slot:cell(preview)="data">
                                <img class="noscaleimg" stye="image-rendering: pixelated;" height="200" :src="`${appConfig.API_URL}/api/download/session/${item.ident}/episode/${selectedEpisodeId}/images/${pad(data.item.episode_step_count,5)}.jpg`" />xx
                            </template>
                            <template v-slot:cell(test)="data">
                                {{data.item}}

                            </template>
                        </b-table>
                        <b-pagination v-model="currentPage" :total-rows="rows" :per-page="perPage" aria-controls="datatable"></b-pagination>
                    </b-card-text>
                </b-card>
            </b-col>
        </b-row>
    </b-container>
</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import gql from "graphql-tag";
import {
    timelength,
    timedelta,
    timepretty
} from "../funcs";
import {
    appConfig
} from "../app.config"

function formatNum(num, prec) {
    if (num == null) {
        return "";
    } else if (isNaN(num)) {
        return "NaN";
    } else {
        return num.toPrecision(prec);
    }
}

function pad(n, width, z) {
    z = z || '0';
    n = n + '';
    return n.length >= width ? n : new Array(width - n.length + 1).join(z) + n;
}
const fields = [{
        key: "episode_step_count",
        label: "Step",
    },
    {
        key: "reward",
        label: "Reward",
        formatter: (v) => {
            return formatNum(v, 3)
        }
    },
    {
        key: "observation",
        label: "Observation"
    },
    {
        key: "preview",
        label: "Preview",
        formatter: (v) => {
            if (v == null || v.includes("ENCODE_FAILED")) {
                return "NA"
            } else {
                return v
            }
        }
    },
    {
        key: "done",
        label: "Done"

    },
    {
        key: "action",
        label: "Action (after observation)"
    },

];

const GET_SESSION = gql `
  query GetSession($ident: String!) {
    session(ident: $ident) {
        id
      ident
      envSpecId
      created
      config
      runInfo
      status
      createdTimestamp
    }
  }
`;

export default {
    name: "Session",
    components: {},
    apollo: {
        // Simple query that will update the 'hello' vue property
        session: {
            query: GET_SESSION,
            variables() {
                return {
                    ident: this.uid,
                };
            },
        },
    },
    data() {
        return {
            appConfig,
            perPage: 10,
            currentPage: 1,
            session: {},
            fields,
            selectedEpisodeId: null,
            fps: 15,
            fpsOptions:[
                { value: 2, text: '2 FPS' },
                { value: 5, text: '5 FPS' },
                { value: 15, text: '15 FPS' },
                { value: 60, text: '60 FPS' },
                { value: 120, text: '120 FPS' },
                { value: 180, text: '180 FPS' }
            ],

            episodes: [],
            episode: [],
            maxEpisode: "",
            message: "",
            imageURL: "placeholder.jpg",
            videoURL: ""

        };
    },
    props: {
        uid: String,
        episodeId: String
    },
    computed: {
        episodeList() {

            if (this.episodes.length == 0)
                return []
            else
                return this.episodes.map(v => {
                    return {
                        text: v,
                        value: v
                    }
                })

        },
        item() {
            return this.session;
        },
        rows() {
            return this.episode.length;
        },
    },
    mounted() {
        //
    },
    methods: {
        pad,
        updateEpisode() {
            this.$router.push({
                path: `/episode/view/${this.uid}`,
                query: {
                    episodeId: this.selectedEpisodeId
                }
            });
            this.loadData()

        },
        timedelta,
        timelength,
        formatNum,
        imageError(event) {
            console.log(event);
            this.imageURL = "placeholder.jpg";
        },

        loadData() {
            console.log(this.uid);
            // ${appConfig.API_URL}/api/session_episode_mp4/${this.uid}/${this.maxEpisode}
            axios
                .get(`${appConfig.API_URL}/api/session_episodes/${this.uid}`)
                .then((response) => {
                    // JSON responses are automatically parsed.
                    const data = response.data["items"]
                    if (data)
                        this.episodes = data; // flattenObject(response.data['sessions'])

                })
                .catch((e) => {
                    this.error = e;
                });

            axios
                .get(`${appConfig.API_URL}/api/session_episode_by_id/${this.uid}/${this.selectedEpisodeId}`)
                .then((response) => {
                    console.log(JSON.stringify(response, null, 2))
                    const data = response.data["item"]
                    const result = []

                    if (data != null) {
                        const keys = Object.keys(data)
                        const size = data[keys[0]].length

                        for (let i = 0; i < size; i++) {
                            const container = {};
                            keys.forEach(k => {
                                container[k] = data[k][i]

                            })
                            result.push(container)
                        }
                    }

                    // JSON resonses are automatically parsed.
                    this.episode = result
                    this.videoURL = `${appConfig.API_URL}/api/session_episode_mp4/${this.uid}/${this.selectedEpisodeId}?fps=${this.fps}`;

                })
                .catch((e) => {
                    this.error = e;
                    console.log(this.error)
                });

        },
    },
    // Fetches posts when the component is created.
    created() {
        this.selectedEpisodeId = this.episodeId

        this.loadData();
    },
    beforeDestroy() {
        //
    },
};
</script>

<style>
/* a.page-link{
   color: black;
 } */
.plot {
    height: 350px;
}

.plotcontainer {
    border-style: solid;
    border-width: 1px;
    border-color: #ccc;
    /* padding: 10px; */
    margin: 4px;
}
</style>
